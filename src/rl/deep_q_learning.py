import pickle
import random
import sys
import xml.etree.ElementTree as ET
from typing import Set, Tuple

import numpy as np
from src.environment.environment import SimEnv
from src.rl.dqn import DQN
from src.rl.logger import Logger
from tqdm import tqdm


class DeepQLearner():
    ACTIONS = [
        [1, 0, 0, 0, 0, 0],     # rotate anchor
        [0, 1, 0, 0, 0, 0],     # tilt module 1
        # [0, 0, 1, 0, 0, 0],     # rotate module 1
        [0, 0, 0, 1, 0, 0],     # tilt module 2
        # [0, 0, 0, 0, 1, 0],     # rotate module 2
        [0, 0, 0, 0, 0, 1],     # tilt module 3

        [-1, 0, 0, 0, 0, 0],    # rotate anchor
        [0, -1, 0, 0, 0, 0],    # tilt module 1
        # [0, 0, -1, 0, 0, 0],    # rotate module 1
        [0, 0, 0, -1, 0, 0],    # tilt module 2
        # [0, 0, 0, 0, -1, 0],    # rotate module 2
        [0, 0, 0, 0, 0, -1]     # tilt module 3
    ]

    WORKSPACE_DISCRETIZATION = 0.2

    def __init__(self, env_path: str, urdf_path: str,
                 use_graphics: bool = False, network_path = "") -> None:
        urdf = ET.tostring(ET.parse(urdf_path).getroot(), encoding='unicode')
        self.env = SimEnv(env_path, urdf, use_graphics=use_graphics)
        self.workspace = set(self._get_workspace())
        self.goal_samples = self.workspace.copy()

        self.dqn = DQN(len(self.ACTIONS), network_path)
        self.training = not network_path

        self.logger = Logger()

    def _generate_actions(self, nr_of_modules):
        pass

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
        with open('../environment/robot_workspace.pkl', "rb") as file:
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
        for episode in tqdm(range(num_episodes), desc='Deep Q-Learning'):
            # the end effector position is already randomized after reset()
            observations = self.env.reset()

            goal = self._generate_goal()
            self.env.set_goal((0,goal[0],goal[1]))
            # print(goal)
            # print(observations[13:15])

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
                    self.dqn.update(state, new_state,action_index, reward, finished)

                episode_step += 1
                state = new_state

            self.logger.log_episode(episode, state, goal, episode_step, total_finished)

        self.env.close()

    def predict(self, state: np.ndarray, stochastic: bool = False) -> int:
        if stochastic and np.random.rand() < self.dqn.eps:
            return np.random.randint(len(self.ACTIONS))
        return self.dqn.lookup(state)

if __name__ == "__main__":

    ENV_PATH =  "../../build/simenv.x86_64"
    URDF_PATH = "../environment/robot.urdf"

    if len(sys.argv) == 2:
        model = DeepQLearner(ENV_PATH, URDF_PATH, False, sys.argv[1])
    else:
        model = DeepQLearner(ENV_PATH, URDF_PATH, True)

    model.learn()
