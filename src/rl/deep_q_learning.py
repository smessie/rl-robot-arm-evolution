import random
import signal
import sys
import xml.etree.ElementTree as ET
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from configs.walls import (WALL_9x9_GAP_3x3_BOTTOM_LEFT,
                           WALL_9x9_GAP_3x3_BOTTOM_LEFT_CENTER_COORD,
                           WALL_9x9_GAP_3x3_BOTTOM_RIGHT,
                           WALL_9x9_GAP_3x3_BOTTOM_RIGHT_CENTER_COORD,
                           WALL_9x9_GAP_3x3_TOP_LEFT,
                           WALL_9x9_GAP_3x3_TOP_LEFT_CENTER_COORD,
                           WALL_9x9_GAP_3x3_TOP_RIGHT,
                           WALL_9x9_GAP_3x3_TOP_RIGHT_CENTER_COORD)
from environment.environment import SimEnv
from morphevo.workspace import Workspace
from rl.dqn import DQN
from rl.logger import Logger
from coevolution.arm import Arm
from util.config import get_config

class DeepQLearner:
    """! The Deep Q learner class.
    Defines the class that will learn based on a Deep-Q Network.
    """

    def __init__(self, env_path: str, urdf_path: str = None, urdf: str = None,
                 use_graphics: bool = False, network_path="") -> None:
        """! The DeepQLearner class initializer.
        @param env_path Path of the environment executable.
        @param urdf_path Path to the robot urdf file.
        @param urdf Instance that represents the robot urdf.
        @param use_graphics Boolean that turns graphics on or off.
        @param network_path Path to the Deep-Q network that should be used.
        @return  An instance of the DeepQLearner class.
        """

        if urdf_path:
            urdf = ET.tostring(ET.parse(urdf_path).getroot(), encoding='unicode')
        assert urdf is not None, "Error: No urdf given."

        parameters = get_config()
        self.goal_ball_diameter = parameters.goal_bal_diameter

        self.use_walls = parameters.use_walls
        self.env = SimEnv(env_path, str(urdf), use_graphics=use_graphics)

        workspace = Workspace(*parameters.workspace_parameters)
        self.x_range = workspace.get_x_range()
        self.y_range = workspace.get_y_range()
        self.z_range = workspace.get_z_range()

        self.env.set_workspace((*workspace.cube_offset, workspace.side_length))

        self.actions = self.get_action_space(self.env.joint_amount)
        self.dqn = self.make_dqn(network_path)

        self.training = not network_path

        self.logger = Logger()

        if self.use_walls:
            self.walls = [  WALL_9x9_GAP_3x3_TOP_LEFT, WALL_9x9_GAP_3x3_TOP_RIGHT,
                            WALL_9x9_GAP_3x3_BOTTOM_LEFT, WALL_9x9_GAP_3x3_BOTTOM_RIGHT]
            self.wall_centers = [   WALL_9x9_GAP_3x3_TOP_LEFT_CENTER_COORD,
                                    WALL_9x9_GAP_3x3_TOP_RIGHT_CENTER_COORD,
                                    WALL_9x9_GAP_3x3_BOTTOM_LEFT_CENTER_COORD,
                                    WALL_9x9_GAP_3x3_BOTTOM_RIGHT_CENTER_COORD]
            self.current_wall_index = 0

    def handler(self, *_):
        if self.training:
            res = input("Ctrl-c was pressed. Do you want to save the DQN? (y/n) ")
            if res == 'y':
                self.save()
            sys.exit(1)

    def save(self, path = "./rl/networks/most_recently_saved_network.pkl"):
        """! Save the trained network in a pickle file
        @param path Path to file where the network will be saved.
        """
        self.dqn.save(path)

    def get_action_space(self, number_of_joints) -> np.ndarray:
        """! Get all possible actions given the amount of joints
        @param number_of_joints The amount of joints the robot arm has.
                A complex module has 2 joints, rotating and tilting.
        @return A list of the actions
        """
        actions = np.identity(number_of_joints)
        return np.concatenate([actions, (-1)*actions])

    def get_new_wall(self) -> Tuple[int, np.ndarray, Tuple]:
        """! Get one of the possible walls at random
        @return A wall.
        """
        wall_index = random.randint(0, len(self.walls)-1)
        return wall_index, self.walls[wall_index], self.wall_centers[wall_index]

    def make_dqn(self, network_path="") -> DQN:
        """! Create an instance of the DQN.
        @param network_path Path to network, if the goal is testing, not training.
        @return A DQN instance.
        """
        return DQN(len(self.actions), state_size=9 if self.use_walls else 6, network_path=network_path)

    def _generate_goal(self) -> np.ndarray:
        """! Generate a goal inside the goal space
        @return A goal.
        """
        goal = []
        for axis_range in [self.x_range, self.y_range, self.z_range]:
            range_size = axis_range[1] - axis_range[0]
            goal.append(random.random() * range_size + axis_range[0])
        return np.array(goal)

    def _calculate_state(self, observations: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """! Calculate the current state
        @param observations Observations used to calculate the state.
        @param goal Goal used to calculate the state.
        @return The state.
        """
        ee_pos = self._get_end_effector_position(observations)

        # return np.array([*ee_pos, *observations[:self.joint_amount * 4], *goal], dtype=float)
        if self.use_walls:
            return np.array([*ee_pos, *goal, *self.wall_centers[self.current_wall_index]], dtype=float)
        return np.array([*ee_pos, *goal], dtype=float)

    def _get_end_effector_position(self, observations: np.ndarray) -> np.ndarray:
        """! Get the end effector position
        @param observations Observations to extract the end effector position from
        @return End effector position.
        """
        return observations[self.env.joint_amount * 4:self.env.joint_amount * 4 + 3]

    def _calculate_reward(self, previous_position: np.ndarray, new_position: np.ndarray, goal: np.ndarray) \
            -> Tuple[float, bool]:
        """! Calculate the reward
        @param previous_position Previous position the end effector was in.
        @param new_position New position the end effector is in.
        @param goal Goal the end effector is trying to reach.
        @return The reward and if the goal was reached.
        """
        prev_distance_from_goal = np.linalg.norm(previous_position - goal)
        new_distance_from_goal = np.linalg.norm(new_position - goal)

        if new_distance_from_goal <= self.goal_ball_diameter:
            return 30, True
        moved_distance = prev_distance_from_goal - new_distance_from_goal

        return 12 * moved_distance, False

    def step(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """! Move 1 step forward in the simulation
        @param state Current staten.
        @return The action that was taken and the observations that were made.
        """
        action_index = self.predict(state)
        action = np.array(self.actions[action_index])
        # Execute the action in the environment
        observations = self.env.step(action)
        return action_index, observations

    def learn(self, number_of_episodes: int = 10000, steps_per_episode: int = 1000, logging: bool = False) -> float:
        """! The learning loop of the reinforcement learning part.
        @param number_of_episodes Maximum amount of episodes.
        @param steps_per_episode Maximum amount of steps each episode.
        @param logging If true there will be wandb logs
        @return Success rate throughout training.
        """
        episodes_finished = [False] * 50
        total_finished = 0
        for episode in tqdm(range(number_of_episodes), desc='Deep Q-Learning'):
            # the end effector position is already randomized after reset()
            observations = self.env.reset()
            if self.use_walls:
                self.current_wall_index, new_wall, _ = self.get_new_wall()
                self.env.replace_walls(new_wall)

            goal = self._generate_goal()
            self.env.set_goal(tuple(goal))

            state = self._calculate_state(observations, goal)
            previous_position = self._get_end_effector_position(observations)
            episode_step = 0
            finished = False
            while not finished and episode_step < steps_per_episode:
                # Get an action and execute
                action_index, observations = self.step(state)

                new_state = self._calculate_state(observations, goal)

                # Calculate reward
                new_position = self._get_end_effector_position(observations)
                reward, finished = self._calculate_reward(
                    previous_position, new_position, goal)
                previous_position = new_position  # this is not in the state, but is useful for reward calculation

                # network update
                if self.training:
                    self.dqn.update(state, new_state, action_index, reward, finished)

                episode_step += 1
                state = new_state

            if finished:
                total_finished += 1
            episodes_finished = episodes_finished[1:] + [finished]

            if logging:
                self.logger.log_episode(episode, state, goal, episode_step, total_finished,
                                        episodes_finished, reward, self.dqn.eps)

        if self.training:
            self.save()

        self.env.close()
        return total_finished/number_of_episodes

    def predict(self, state: np.ndarray) -> int:
        """! Take an action
        @param state Current state.
        @return The chosen action.
        """
        if self.training and np.random.rand() < self.dqn.eps:
            return np.random.randint(len(self.actions))
        # The testing also needs some randomness
        if not self.training and np.random.rand() < 0.2:
            return np.random.randint(len(self.actions))
        action = self.dqn.get_best_action(state)
        return action

    def get_score(self, number_of_joints: int, workspace: Workspace, episodes: int = 200) -> float:
        """! Run a training cycle on a robot arm and workspace.
        @param number_of_joints The amount of joints the robot we run the cycle for has
        @param workspace The goal workspace the robot needs to work on.
        @param episodes Amount of episodes the training cycle needs to do.
        @return The success rate of the training.
        """
        self.x_range = workspace.get_x_range()
        self.y_range = workspace.get_y_range()
        self.z_range = workspace.get_z_range()

        self.actions = self.get_action_space(number_of_joints)
        self.dqn = self.make_dqn()
        return self.learn(number_of_episodes=episodes)

def rl(network_path=""):
    """! Run reinforcement learning
    @param network_path The path to a network that is passed when the training has been done and we want to test.
    """
    config = get_config()
    if network_path:
        model = DeepQLearner(env_path=config.path_to_unity_executable,
                            urdf_path=config.path_to_robot_urdf,
                            use_graphics=config.rl_use_graphics_testing,
                            network_path=network_path)
    else:
        model = DeepQLearner(env_path=config.path_to_unity_executable,
                            urdf_path=config.path_to_robot_urdf,
                            use_graphics=config.rl_use_graphics_training)

    signal.signal(signal.SIGINT, model.handler)
    model.learn(logging=True)

def train(arms: List[Arm]) -> List[Arm]:
    """! Run a reinforcement learning training cycle on each given arm.
    @param arms The list of arms.
    @return The arms with their reinforcement learning model and success rate added.
    """
    config = get_config()
    for arm in arms:
        model = DeepQLearner(env_path=config.path_to_unity_executable,
                             urdf=arm.urdf,
                             use_graphics=config.rl_use_graphics_training)

        arm.success_rate = model.learn(
            number_of_episodes=config.episodes, steps_per_episode=config.steps_per_episode, logging=False
        )
        arm.rl_model = model.dqn

    return arms
