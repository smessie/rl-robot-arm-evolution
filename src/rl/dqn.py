import random
from collections import deque

import numpy as np
import torch


class RobotNetwork(torch.nn.Module):
    def __init__(self, hidden_nodes, number_of_actions, state_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(state_size, hidden_nodes)
        self.linear2 = torch.nn.Linear(hidden_nodes, 32)
        self.linear3 = torch.nn.Linear(32, number_of_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        return self.linear3(x)

class DQN:
    GAMMA = 0.99
    EPS_END = 0.1
    EPS_DECAY = 0.999995
    BATCH_SIZE = 64
    MEM_SIZE = 1000

    def __init__(self, n_actions: int, state_size, network_path = ""):
        self.eps = 1

        if network_path:
            self.network = torch.load(network_path)
            self.network.eval()
        else:
            self.network = RobotNetwork(64, n_actions, state_size)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        self.memory = deque(maxlen=self.MEM_SIZE)


    def update(self, _state, new_state, action, reward, finished):
        '''
        Method for updating the network
        Parameters:
            _state: [int]
        '''
        state = torch.tensor([_state], dtype=torch.float)
        next_state = torch.tensor([new_state], dtype=torch.float)

        # Store the transition in memory
        self.memory.append((state, action, reward, next_state, int(finished)))

        # Move to the next state
        state = next_state

        # Experience replay
        if len(self.memory) >= self.BATCH_SIZE:
            batch = random.sample(self.memory, self.BATCH_SIZE)
            states, actions, rewards, n_states, dones = zip(*batch)
            state_batch = torch.cat(states,0)
            action_batch = torch.tensor(actions)
            reward_batch = torch.tensor(rewards)
            n_states = torch.cat(n_states)
            dones = torch.tensor(dones)

            # EXPERIENCE REPLAY

            # Bereken de Q-values voor de gegeven toestanden
            curr_Q = self.network(state_batch.float()).gather(1, action_batch.unsqueeze(1))
            curr_Q = curr_Q.squeeze(1)

            # Bereken de Q-values voor de volgende toestanden (n_states)
            max_next_Q = (1-dones) * self.network(n_states.float()).max(1)[0].detach()

            # Gebruik deze Q-values om targets te berekenen
            targets = reward_batch + (self.GAMMA*max_next_Q)

            # Bereken de loss
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(curr_Q, targets.float())
            self.optimizer.zero_grad()
            loss.backward()

            # Voer een optimalisatiestap uit
            self.optimizer.step()

            # Decay exploration rate
            self.eps *= self.EPS_DECAY
            self.eps = max(self.EPS_END, self.eps)

    def lookup(self, state: np.ndarray) -> int:
        with torch.no_grad():
            x = self.network(torch.tensor([state], dtype=torch.float))
            probs, indices = torch.topk(x, 1)
            return indices[0].item()
            #for action in indices[0]:
                #move = self.action_to_move(self.output_to_action(action.item()), self.possible_moves)
                #if not move is None:
                    #return action.item()

#                return self.network(state).argmax().item()

    def save(self, path: str):
        torch.save(self.network, path)
