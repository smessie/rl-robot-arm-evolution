import pickle
import random
import signal
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

    WORKSPACE_DISCRETIZATION = 0.2

    # hyperparameters
    EPSILON = 0.1
    ALPHA = 0.1
    GAMMA = 0.99

    def __init__(self, env_path: str, urdf_path: str,
                 use_graphics: bool = False, filename = "") -> None:
        urdf = ET.tostring(ET.parse(urdf_path).getroot(), encoding='unicode')
        self.env = SimEnv(env_path, urdf, use_graphics=use_graphics)
        self.workspace = set(self._get_workspace())
        self.goal_samples = self.workspace.copy()
        self.testing = False

        if filename != "":
            with open(filename, 'rb') as file:
                self.q_table = pickle.load(file)
            self.testing = True
        else:
            self.q_table = QTable(len(self.workspace) ** 2,
                                len(self.ACTIONS),
                                self.ALPHA,
                                self.GAMMA)

        if self.testing:
            self.q_table.visualize()
        self.logger = Logger()

    def handler(self, *_):
        self.q_table.visualize()
        if not self.testing:
            res = input("Ctrl-c was pressed. Do you want to save the QTable? (y/n) ")
            if res == 'y':
                self.save()
                sys.exit(1)

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

        return np.array([ee_pos[0], ee_pos[1], goal[0], goal[1]], dtype=int)

    def _calculate_reward(self, prev_absolute_pos: np.ndarray, new_absolute_pos: np.ndarray,
                          goal: np.ndarray) -> Tuple[float, bool]:
        prev_distance_from_goal = np.linalg.norm(prev_absolute_pos - goal)
        new_distance_from_goal = np.linalg.norm(new_absolute_pos - goal)

        if new_distance_from_goal <= 2*self.WORKSPACE_DISCRETIZATION:
            return 10, True

        return prev_distance_from_goal - new_distance_from_goal, False

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
                # Get an action
                action_index = self.predict(state, stochastic=(not self.testing))
                actions = np.array(self.ACTIONS[action_index])

                # Execute the action in the environmenthttps://wandb.ai/selab/sel3-rl-tutorial/runs/lvq90jw0
                observations = self.env.step(actions)

                new_state = self._calculate_state(observations, goal)

                # Calculate reward
                new_absolute_pos = self._discretize_position(observations[13:15])
                reward, finished = self._calculate_reward(
                    prev_absolute_pos, new_absolute_pos, goal)
                prev_absolute_pos = new_absolute_pos  # this is not in the state, but is useful for reward calculation

                if finished:
                    total_finished += 1

                # QTable update
                if not self.testing:
                    self.q_table.update(state, new_state, action_index, reward)

                episode_step += 1
                state = new_state

            self.logger.log_episode(episode, state, goal, episode_step, finished, total_finished,self.q_table)

        self.env.close()
        if not self.testing:
            self.save()

    def predict(self, state: np.ndarray, stochastic: bool = False) -> int:
        if stochastic and np.random.rand() < self.EPSILON:
            return np.random.randint(len(self.ACTIONS))
        return self.q_table.lookup(state)

    def save(self):
        with open('q_tables/q_table.pkl', 'wb') as file:
            pickle.dump(self.q_table, file, protocol=pickle.HIGHEST_PROTOCOL)

    def test(self, filename, max_steps: int = 500):
        with open(filename, 'rb') as file:
            self.q_table = pickle.load(file)

        for i, goal in enumerate(self.workspace):
            print(f"Going to goal {i}")
            goal = np.array(goal)
            observations = self.env.reset()
            state = self._calculate_state(observations, goal)
            prev_absolute_pos = self._discretize_position(observations[13:15])

            episode_step = 0
            finished = False
            while not finished and episode_step < max_steps:

                # Get an action
                action_index = self.predict(state, stochastic=False)
                actions = np.array(self.ACTIONS[action_index])

                # Execute the action in the environment
                observations = self.env.step(actions)
                new_state = self._calculate_state(observations, goal)

                # Calculate reward
                new_absolute_pos = self._discretize_position(observations[13:15])
                _, finished = self._calculate_reward(
                    prev_absolute_pos, new_absolute_pos, goal)
                prev_absolute_pos = new_absolute_pos  # this is not in the state, but is useful for reward calculation

                episode_step += 1
                state = new_state

            self.logger.log_test(
                episode_step, prev_absolute_pos, goal, i)

        self.env.close()


if __name__ == "__main__":
    # execute test loop by adding file to read qtable from
    # without file it will train

    ENV_PATH = "src/environment/unity_environment_tutorial/simenv.x86_64"
    URDF_PATH = "src/environment/robot_tutorial.urdf"

    if len(sys.argv) == 2:
        model = QLearner(ENV_PATH, URDF_PATH, True, sys.argv[1])
    else:
        model = QLearner(ENV_PATH, URDF_PATH, False)

    signal.signal(signal.SIGINT, model.handler)
    model.learn()
