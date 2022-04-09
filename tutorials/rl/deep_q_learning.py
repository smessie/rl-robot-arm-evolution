import torch
import random
from itertools import count
from collections import deque
from random import randint
from termcolor import colored
import sys
import pickle
import random
import sys
import xml.etree.ElementTree as ET
from typing import Set, Tuple

import numpy as np
from src.environment.environment import SimEnv
from src.rl.logger import Logger
from src.rl.q_table import QTable
from tqdm import tqdm


class RobotNetwork(torch.nn.Module):
    def __init__(self, hidden_nodes, number_of_actions):
        super(RobotNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(4, hidden_nodes)
        self.linear2 = torch.nn.Linear(hidden_nodes, hidden_nodes)
        self.linear3 = torch.nn.Linear(hidden_nodes, number_of_actions)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        return self.linear3(x)


class DeepQLearner():
    ACTIONS = [
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ]

    WORKSPACE_DISCRETIZATION = 0.2

    def __init__(self, env_path: str, urdf_path: str,
                 use_graphics: bool = False, network_path = "") -> None:
        urdf = ET.tostring(ET.parse(urdf_path).getroot(), encoding='unicode')
        self.env = SimEnv(env_path, urdf, use_graphics=use_graphics)
        self.workspace = set(self._get_workspace())
        self.goal_samples = self.workspace.copy()

        self.BATCH_SIZE = 64 
        self.MEM_SIZE = 1000
        self.GAMMA = 0.99
        self.EPS_END = 0.05
        self.EPS_DECAY = 0.9999
        self.eps = 1.0
        self.N_ACTIONS = 4 #todo: should not be hard coded 

        if network_path:
            self.network = torch.load(network_path)
            self.network.eval()
            self.training = False
        else:
            self.network = RobotNetwork(64, self.N_ACTIONS)
            self.training = True

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        self.memory = deque(maxlen=self.MEM_SIZE)

        self.normalized = True
        self.logger = Logger()

    def _discretize_position(self, pos: np.ndarray) -> np.ndarray:
        discretized_pos = (pos / self.WORKSPACE_DISCRETIZATION).astype(int)
        return discretized_pos

    def _discretize_direction(self, pos: np.ndarray, goal: np.ndarray):
        direction = goal - pos
        result = [0,0]
        if direction[0] != 0:
            result[0] = direction[0] / np.abs(direction[0])
        if direction[1] != 0:
            result[1] = direction[1] / np.abs(direction[1])

        return result

    def _get_workspace(self) -> Set[Tuple[float, float]]:
        with open('src/environment/robot_workspace.pkl', "rb") as file:
            workspace = pickle.load(file)

        workspace = {tuple(self._discretize_position(np.array(pos)))
                     for pos in workspace}
        return workspace

    def _generate_goal(self) -> np.ndarray:
        if len(self.goal_samples) == 0:
            self.goal_samples = self.workspace.copy()

        goal = random.sample(self.goal_samples, k=1)[0]
        self.goal_samples.remove(goal)

        return np.array(goal)

    def _calculate_state(self, observations: np.ndarray,
                         goal: np.ndarray) -> np.ndarray:
        # [j0, j0x, j0y, j0z, j1, j1x, j1y, j1z,
        #  j2, j2x, j2y, j2z, ee_x, ee_y, ee_z]
        # [EEPOS, GOAL_y, GOAL_z]
        ee_pos = observations[13:15]

        ee_pos = self._discretize_position(ee_pos)
        goal = self._discretize_direction(ee_pos, goal)

        return np.array([ee_pos[0], ee_pos[1], goal[0], goal[1]], dtype=float)

    def _calculate_reward(self, prev_absolute_pos: np.ndarray, new_absolute_pos: np.ndarray,
                          goal: np.ndarray) -> Tuple[float, bool]:
        prev_distance_from_goal = np.linalg.norm(prev_absolute_pos - goal)
        new_distance_from_goal = np.linalg.norm(new_absolute_pos - goal)

        if new_distance_from_goal <= 2*self.WORKSPACE_DISCRETIZATION:
            return 10, True

        return prev_distance_from_goal - new_distance_from_goal, False
    
    def step(self, state):
        action_index = self.predict(state, stochastic=(self.training))
        actions = np.array(self.ACTIONS[action_index])

        # Execute the action in the environment
        observations = self.env.step(actions)
        return action_index,observations

    def learn(self, num_episodes: int = 10000,
              steps_per_episode: int = 500) -> None:

        total_finished = 0
        for episode in tqdm(range(num_episodes), desc='Q-Learning'):
            # the end effector position is already randomized after reset()
            observations = self.env.reset()

            goal = self._generate_goal()

            state = self._calculate_state(observations, goal)
            prev_absolute_pos = self._discretize_position(observations[13:15])

            episode_step = 0
            finished = False
            while not finished and episode_step < steps_per_episode:
                # Get an action and execute
                action_index, observations = self.step(state)

                new_state = self._calculate_state(observations, goal)

                # Calculate reward
                new_absolute_pos = self._discretize_position(observations[13:15])
                reward, finished = self._calculate_reward(
                    prev_absolute_pos, new_absolute_pos, goal)
                prev_absolute_pos = new_absolute_pos  # this is not in the state, but is useful for reward calculation

                if finished:
                    total_finished += 1

                # network update
                if self.training:
                    self.update(state, new_state,action_index, reward, finished)

                episode_step += 1
                state = new_state

            self.logger.log_episode(episode, state, goal, episode_step, total_finished)

        self.env.close()

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
                
    def predict(self, state: np.ndarray, stochastic: bool = False) -> int:
        if stochastic and np.random.rand() < self.eps:
            with torch.no_grad():
                x = self.network(torch.tensor([state], dtype=torch.float))
                probs, indices = torch.topk(x, 1)
                return indices[0].item()
                #for action in indices[0]:
                    #move = self.action_to_move(self.output_to_action(action.item()), self.possible_moves)
                    #if not move is None:
                        #return action.item()

#                return self.network(state).argmax().item()
        else:
            return np.random.randint(len(self.ACTIONS))

if __name__ == "__main__":

    ENV_PATH = "src/environment/unity_environment_tutorial/simenv.x86_64"
    URDF_PATH = "src/environment/robot_tutorial.urdf"

    if len(sys.argv) == 2:
        model = DeepQLearner(ENV_PATH, URDF_PATH, False, sys.argv[1])
    else:
        model = DeepQLearner(ENV_PATH, URDF_PATH, False)

    model.learn()