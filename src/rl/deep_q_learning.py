import random
from shutil import move
import signal
import sys
import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np
from tqdm import tqdm

from configs.env import PATH_TO_UNITY_EXECUTABLE
from configs.walls import WALL_9x9_GAP_3x3, WALL_9x9_GAP_9x3
from environment.environment import SimEnv
from rl.dqn import DQN
from rl.logger import Logger


class DeepQLearner:
    ACTIONS = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [-1, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, -1],
    ]

    WORKSPACE_DISCRETIZATION = 0.2
    GOAL_BAL_DIAMETER = 0.6

    def __init__(self, env_path: str, urdf_path: str,
                 use_graphics: bool = False, network_path = "") -> None:
        urdf = ET.tostring(ET.parse(urdf_path).getroot(), encoding='unicode')
        self.env = SimEnv(env_path, urdf, use_graphics=use_graphics)

        self.amount_of_modules = self.env.joint_amount

        self.x_range = [-8,8]
        self.y_range = [3,6]
        self.z_range = [15,18]

        # state_size is 6: 3 coords for the end effector position, 3 coords for the goal
        # self.dqn = DQN(len(self.ACTIONS), state_size=6 + self.amount_of_modules * 4, network_path=network_path)
        self.dqn = DQN(len(self.ACTIONS), state_size=6, network_path=network_path)

        self.training = not network_path
        self.penalty = 0

        self.logger = Logger()

    def handler(self, *_):
        if self.training:
            res = input("Ctrl-c was pressed. Do you want to save the QTable? (y/n) ")
            if res == 'y':
                self.save()
                sys.exit(1)

    def save(self):
        self.dqn.save("./networks/most_recently_saved_network.pkl")

    def _generate_actions(self, nr_of_modules):
        pass

    def _calculate_direction(self, pos: np.ndarray, goal: np.ndarray):
        direction = goal - pos

        result = [0,0,0]
        for i, axis_direction in enumerate(direction):
            if axis_direction != 0:
                result[i] = axis_direction / np.abs(axis_direction)

        return result

    def _generate_goal(self) -> np.ndarray:
        goal = []
        for axis_range in [self.x_range, self.y_range, self.z_range]:
            range_size = axis_range[1] - axis_range[0]
            goal.append(random.random()*range_size + axis_range[0])
        return np.array(goal)

    def _calculate_state(self, observations: np.ndarray,
                         goal: np.ndarray) -> np.ndarray:
        # [j0, j0x, j0y, j0z, j1, j1x, j1y, j1z,
        #  j2, j2x, j2y, j2z, ee_x, ee_y, ee_z]
        # [EEPOS, GOAL_y, GOAL_z]
        ee_pos = self._get_end_effector_position(observations)

        # return np.array([*ee_pos, *observations[:self.amount_of_modules * 4], *goal], dtype=float)
        return np.array([*ee_pos, *goal], dtype=float)

    def _get_end_effector_position(self, observations: np.ndarray):
        return observations[self.amount_of_modules * 4:self.amount_of_modules * 4 + 3]

    def _calculate_reward(self, prev_pos: np.ndarray, new_pos: np.ndarray,
                          goal: np.ndarray) -> Tuple[float, bool]:
        prev_distance_from_goal = np.linalg.norm(prev_pos - goal)
        new_distance_from_goal = np.linalg.norm(new_pos - goal)

        if new_distance_from_goal <= self.GOAL_BAL_DIAMETER:
            return 20, True
        
        moved_distance = prev_distance_from_goal - new_distance_from_goal

        if moved_distance < 0.2:
            self.penalty = min(self.penalty + 0.2, 5)
        else:
            self.penalty = max(self.penalty - 0.2, 0)

        return 10*(moved_distance) - self.penalty, False

    def step(self, state):
        action_index = self.predict(state, stochastic=self.training)
        actions = np.array(self.ACTIONS[action_index])

        # Execute the action in the environment
        observations = self.env.step(actions)
        return action_index, observations

    def learn(self, num_episodes: int = 10000,
              steps_per_episode: int = 500) -> None:

        total_finished = 0
        for episode in tqdm(range(num_episodes), desc='Deep Q-Learning'):
            self.penalty = 0
            # the end effector position is already randomized after reset()
            observations = self.env.reset()

            goal = self._generate_goal()
            self.env.set_goal(tuple(goal))

            state = self._calculate_state(observations, goal)
            prev_pos = self._get_end_effector_position(observations)
            episode_step = 0
            finished = False
            while not finished and episode_step < steps_per_episode:
                # Get an action and execute
                action_index, observations = self.step(state)

                new_state = self._calculate_state(observations, goal)

                # Calculate reward
                new_pos = self._get_end_effector_position(observations)
                reward, finished = self._calculate_reward(
                    prev_pos, new_pos, goal)
                prev_pos = new_pos  # this is not in the state, but is useful for reward calculation

                if finished:
                    total_finished += 1

                # network update
                if self.training:
                    self.dqn.update(state, new_state,action_index, reward, finished)

                episode_step += 1
                state = new_state

            self.logger.log_episode(episode, state, goal, episode_step, total_finished, reward, self.dqn.eps)

        self.save()
        self.env.close()

    def predict(self, state: np.ndarray, stochastic: bool = False) -> int:
        if stochastic and np.random.rand() < self.dqn.eps:
            return np.random.randint(len(self.ACTIONS))
        return self.dqn.lookup(state)


def start_rl():
    env_path = PATH_TO_UNITY_EXECUTABLE
    urdf_path = "environment/robot.urdf"

    if len(sys.argv) == 3:
        model = DeepQLearner(env_path, urdf_path, use_graphics=True, network_path=sys.argv[2])
    else:
        model = DeepQLearner(env_path, urdf_path, use_graphics=True)

    model.env.build_wall(WALL_9x9_GAP_9x3)
    signal.signal(signal.SIGINT, model.handler)
    model.learn(steps_per_episode=1000)
