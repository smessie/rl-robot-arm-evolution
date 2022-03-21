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


class QLearner:
    ACTIONS = [
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ]

    WORKSPACE_DISCRETIZATION = 0.5

    # hyperparameters
    EPSILON = 0.1
    ALPHA = 0.1
    GAMMA = 0.99

    def __init__(self, env_path: str, urdf_path: str, use_graphics: bool = False, filename: str = "") -> None:
        urdf = ET.tostring(ET.parse(urdf_path).getroot(), encoding='unicode')
        self.env = SimEnv(env_path, urdf, use_graphics=use_graphics)
        self.workspace = self._get_workspace()
        self.testing = False

        if filename != "":
            print("loading qtable")
            with open(filename, 'rb') as file:
                self.q_table = pickle.load(file)
            self.testing = True
        else:
            self.q_table = QTable(len(self.workspace) ** 2,
                                len(self.ACTIONS),
                                self.ALPHA,
                                self.GAMMA)

        self.logger = Logger()

    def _discretize_position(self, pos: np.ndarray) -> np.ndarray:
        discretized_pos = (pos / self.WORKSPACE_DISCRETIZATION).astype(int)
        return discretized_pos

    def _get_workspace(self) -> Set[Tuple[float, float]]:
        with open ('src/environment/robot_workspace.pkl', "rb") as f:
            workspace = pickle.load(f)

        workspace = {tuple(self._discretize_position(np.array(pos))) for pos in workspace}
        return workspace

    def _generate_goal(self) -> np.ndarray:
        return np.array(random.sample(self.workspace, k=1)[0])

    def _calculate_state(self, observations: np.ndarray, goal: np.ndarray) -> np.ndarray:
        # [j0, j0x, j0y, j0z, j1, j1x, j1y, j1z, j2, j2x, j2y, j2z, ee_x, ee_y, ee_z]
        # [EEPOS, GOAL_y, GOAL_z]
        ee_pos = observations[13:15]

        ee_pos = self._discretize_position(ee_pos)
        return np.array([ee_pos[0], ee_pos[1], goal[0], goal[1]], dtype=int)

    def _calculate_reward(self, prev_state: np.ndarray, new_state: np.ndarray, goal: np.ndarray) -> Tuple[float, bool]:
        prev_distance_from_goal = np.linalg.norm(prev_state[:2] - goal)
        new_distance_from_goal = np.linalg.norm(new_state[:2] - goal)

        if new_distance_from_goal == 0:
            return 1000, True

        return prev_distance_from_goal - new_distance_from_goal, False

    def learn(self, num_episodes: int = 1000, steps_per_episode: int = 500) -> None:
        for episode in tqdm(range(num_episodes), desc='Q-Learning'):
            observations = self.env.reset()
            goal = self._generate_goal()
            state = self._calculate_state(observations, goal)

            episode_step = 0
            finished = False
            while not finished and episode_step < steps_per_episode:
                # Get an action
                action_index = self.predict(state, stochastic=(not self.testing))
                actions = np.array(self.ACTIONS[action_index])

                # Execute the action in the environment
                observations = self.env.step(actions)
                new_state = self._calculate_state(observations, goal)
                # Calculate reward
                reward, finished =  self._calculate_reward(state, new_state, goal)

                # QTable update
                if not self.testing:
                    self.q_table.update(state, new_state, action_index, reward)

                episode_step += 1
                state = new_state

            self.logger.log_episode(episode, state, goal, episode_step, self.q_table)

        self.env.close()
        if not self.testing:
            self.save()

    def predict(self, state: np.ndarray, stochastic: bool = False) -> int:
        if stochastic and np.random.rand() < self.EPSILON:
            return np.random.randint(len(self.ACTIONS))
        return self.q_table.lookup(state)

    def save(self):
        with open('q_tables/q_table_tutorial_3k_2.pkl', 'wb') as file:
            pickle.dump(self.q_table, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    ENV_PATH = "src/environment/unity_environment/simenv.x86_64"
    URDF_PATH = "src/environment/robot.urdf"

    if len(sys.argv) == 2:
        model = QLearner(ENV_PATH, URDF_PATH, False, sys.argv[1])
    else:
        model = QLearner(ENV_PATH, URDF_PATH, False)

    model.learn()
