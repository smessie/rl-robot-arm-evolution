import random
from collections import deque

import numpy as np
import torch

from util.config import get_config


class RobotNetwork(torch.nn.Module):
    """! The Robot Network class.
    Defines the class that will represent the network that is used for deep Q-learning.
    """
    def __init__(self, hidden_nodes: int, number_of_actions: int, state_size: int):
        """! The RobotNetwork class initializer.
        @param hidden_nodes The number of hidden nodes for the hidden layers.
        @param number_of_actions The number of possible actions, important for nodes needed as output.
        @param state_size The number of variables in a state, important for the nodes needed as input.
        @return  An instance of the RobotNetwork class.
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(state_size, hidden_nodes)
        self.linear2 = torch.nn.Linear(hidden_nodes, hidden_nodes)
        self.linear3 = torch.nn.Linear(hidden_nodes, number_of_actions)

    def forward(self, input):
        """! Internal function of the network, used for processing the input of the network. 
        @param input The network input.
        """
        return self.linear3(self.linear2(self.linear1(input)))


class DQN:
    """! The DQN class.
    Defines the class that will represent the Deep Q-Network.
    """
    def __init__(self, number_of_actions: int, state_size: int, network_path=""):
        """! The RobotNetwork class initializer.
        @param number_of_actions The number of hidden nodes for the hidden layers.
        @param state_size The number of variables in a state, important for the nodes needed as input.
        @param network_path The path to the file that contains a (trained) network.
        @return An instance of the DQN class.
        """
        parameters = get_config()
        self.eps = parameters.eps_start
        self.gamma = parameters.gamma
        self.eps_end = parameters.eps_end
        self.eps_decay = parameters.eps_decay
        self.batch_size = parameters.batch_size
        self.mem_size = parameters.mem_size
        self.hidden_nodes = parameters.hidden_nodes

        if network_path:
            self.network = torch.load(network_path)
        else:
            self.network = RobotNetwork(self.hidden_nodes, number_of_actions, state_size)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0001)
        self.memory = deque(maxlen=self.mem_size)

    def _save(self, path: str):
        """! Save the network to a given path. 
        @param path The file path to save the network to. 
        """
        torch.save(self.network, path)

    def _calculate_current_q_value(self, state_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        """! Calculate the current Q-value.
        @param state_batch The states batch. 
        @param action_batch The actions batch. 
        @return Current Q value.
        """
        current_q = self.network(state_batch.float()).gather(1, action_batch.unsqueeze(1))
        return current_q.squeeze(1)

    def _calculate_targets(self, dones: torch.Tensor, next_states: torch.Tensor, reward_batch: torch.Tensor) -> torch.Tensor:
        """! Calculate the targets.
        @param dones Tensor list with finished value or not finished. 
        @param next_states The next states. 
        @param reward_batch The reward batch.
        @return The targets.
        """
        max_next_q = (1 - dones) * self.network(next_states.float()).max(1)[0].detach()

        return reward_batch + (self.gamma * max_next_q)

    def _apply_loss(self, current_q: torch.Tensor, targets: torch.Tensor) -> None:
        """! Apply the loss function.
        @param The current Q-value.
        @param next_states The targets. 
        """
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(current_q, targets.float())
        self.optimizer.zero_grad()
        loss.backward()

    def _experience_replay(self) -> None:
        """! Apply experience replay to the network.
        """
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_batch = torch.cat(states, 0)
        action_batch = torch.tensor(actions)
        reward_batch = torch.tensor(rewards)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones)

        current_q = self._calculate_current_q_value(state_batch, action_batch)
        targets = self._calculate_targets(dones, next_states, reward_batch)
        self._apply_loss(current_q, targets)

        self.optimizer.step()

        self.eps *= self.eps_decay
        self.eps = max(self.eps_end, self.eps)

    # pylint: disable-msg=too-many-locals
    def update(self, state: np.ndarray, next_state: np.ndarray, action: int, reward: float, finished: bool):
        """! Update the network, should happen between each state transition during training.
        @param state The start state of the environment.
        @param next_state The resulting, next state after executing an action.
        @param action The action that was executed.
        @param reward The reward resulting from the action.
        @param finished If the next_state is a finishing state.
        """
        state = torch.tensor([state], dtype=torch.float)
        next_state = torch.tensor([next_state], dtype=torch.float)

        self.memory.append((state, action, reward, next_state, int(finished)))

        state = next_state

        if len(self.memory) >= self.batch_size:
            self._experience_replay()

    def get_best_action(self, state: np.ndarray) -> int:
        """! Look up the best action for a given state. 
        @param state An input state representing the environment. 
        @return An action, represented as a number.
        """
        with torch.no_grad():
            x = self.network(torch.tensor([state], dtype=torch.float))
            _, indices = torch.topk(x, 1)
            return indices[0].item()

