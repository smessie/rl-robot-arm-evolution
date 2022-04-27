import random
import signal
import sys
import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np
from tqdm import tqdm

from configs.env import PATH_TO_UNITY_EXECUTABLE
from environment.environment import SimEnv
from morphevo.workspace import Workspace
from rl.dqn import DQN
from rl.logger import Logger


class DeepQLearner:
    """! The Deep Q learner class.
    Defines the class that will learn based on a Deep-Q Network.
    """

    WORKSPACE_DISCRETIZATION = 0.2
    GOAL_BAL_DIAMETER = 0.6

    def __init__(self, env_path: str, urdf_path: str = None, urdf: str = None,
                 use_graphics: bool = False, network_path="") -> None:
        """! The DeepQLearner class initializer.
        @param env_path Path of the environment executable.
        @param urdf_path [Default = ""] Path to the robot urdf file.
        @param urdf [Default = ""] Instance that represents the robot urdf.
        @param use_graphics [Default = False] Boolean that turns graphics on or off.
        @param network_path [Default = ""] Path to the Deep-Q network that should be used.
        @return  An instance of the DeepQLearner class.
        """

        if urdf_path:
            urdf = ET.tostring(ET.parse(urdf_path).getroot(), encoding='unicode')
        assert urdf is not None, "Error: No urdf given."

        self.env = SimEnv(env_path, str(urdf), use_graphics=use_graphics)
        # todo: niet amount_of_joints
        self.joint_amount = self.env.joint_amount

        self.x_range = [-1.5, 1.5]
        self.y_range = [3, 6]
        self.z_range = [6, 9]

        self.actions = []
        # todo:: dit moet eig met amount_of_joints maar dat geeft maar 1?
        self.set_action_space(4)
        self.dqn = self.make_dqn(network_path)

        self.training = not network_path

        self.logger = Logger()

    def handler(self, *_):
        if self.training:
            res = input("Ctrl-c was pressed. Do you want to save the DQN? (y/n) ")
            if res == 'y':
                self.save()
                sys.exit(1)

    def save(self):
        self.dqn.save("./networks/most_recently_saved_network.pkl")

    def _generate_actions(self, nr_of_modules):
        pass

    def _calculate_direction(self, pos: np.ndarray, goal: np.ndarray):
        direction = goal - pos

        result = [0, 0, 0]
        for i, axis_direction in enumerate(direction):
            if axis_direction != 0:
                result[i] = axis_direction / np.abs(axis_direction)

        return result

    def _generate_goal(self) -> np.ndarray:
        goal = []
        for axis_range in [self.x_range, self.y_range, self.z_range]:
            range_size = axis_range[1] - axis_range[0]
            goal.append(random.random() * range_size + axis_range[0])
        return np.array(goal)

    def _calculate_state(self, observations: np.ndarray,
                         goal: np.ndarray) -> np.ndarray:
        # [j0, j0x, j0y, j0z, j1, j1x, j1y, j1z,
        #  j2, j2x, j2y, j2z, ee_x, ee_y, ee_z]
        # [EEPOS, GOAL_y, GOAL_z]
        ee_pos = self._get_end_effector_position(observations)

        # return np.array([*ee_pos, *observations[:self.joint_amount * 4], *goal], dtype=float)
        return np.array([*ee_pos, *goal], dtype=float)

    def _get_end_effector_position(self, observations: np.ndarray):
        return observations[self.joint_amount * 4:self.joint_amount * 4 + 3]

    def _calculate_reward(self, prev_pos: np.ndarray, new_pos: np.ndarray,
                          goal: np.ndarray) -> Tuple[float, bool]:
        prev_distance_from_goal = np.linalg.norm(prev_pos - goal)
        new_distance_from_goal = np.linalg.norm(new_pos - goal)

        if new_distance_from_goal <= self.GOAL_BAL_DIAMETER:
            return 20, True

        return 10*(prev_distance_from_goal - new_distance_from_goal), False

    def make_dqn(self, network_path=""):
        # state_size is 6: 3 coords for the end effector position, 3 coords for the goal
        # self.dqn = DQN(len(self.actions), state_size=6 + self.joint_amount * 4, network_path=network_path)
        return DQN(len(self.actions), state_size=6, network_path=network_path)

    def set_action_space(self, number_of_joints):
        actions = np.identity(number_of_joints)
        self.actions = np.concatenate([actions, (-1)*actions])

    def step(self, state):
        action_index = self.predict(state, stochastic=self.training)
        actions = np.array(self.actions[action_index])

        # Execute the action in the environment
        observations = self.env.step(actions)
        return action_index, observations

    def learn(self, num_episodes: int = 10000,
              steps_per_episode: int = 1000, logging: bool = True, save: bool = True) -> float:

        total_finished = 0
        for episode in tqdm(range(num_episodes), desc='Deep Q-Learning'):
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
                    self.dqn.update(state, new_state, action_index, reward, finished)

                episode_step += 1
                state = new_state

            if logging:
                self.logger.log_episode(episode, state, goal, episode_step, total_finished, reward, self.dqn.eps)

        if save:
            self.save()

        self.env.close()
        return total_finished/num_episodes

    def predict(self, state: np.ndarray, stochastic: bool = False) -> int:
        if stochastic and np.random.rand() < self.dqn.eps:
            return np.random.randint(len(self.actions))
        return self.dqn.lookup(state)

    def get_score(self, number_of_joints: int, workspace: Workspace):
        self.x_range = workspace.get_x_range()
        self.y_range = workspace.get_y_range()
        self.z_range = workspace.get_z_range()

        self.set_action_space(number_of_joints)
        self.dqn = self.make_dqn()
        return self.learn(num_episodes=200, logging=False, save=False)

def start_rl():
    env_path = PATH_TO_UNITY_EXECUTABLE
    urdf_path = "environment/robot.urdf"

    if len(sys.argv) == 3:
        model = DeepQLearner(env_path, urdf_path=urdf_path, use_graphics=True, network_path=sys.argv[2])
    else:
        model = DeepQLearner(env_path, urdf_path=urdf_path)

    signal.signal(signal.SIGINT, model.handler)
    model.learn()
