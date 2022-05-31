##
# @file
# Contains the SimEnv class, an interface to the Unity project following gym Env interface
# and a function test_environment.
# The function starts the Unity project and tests some basic things like actions and side channels
#
import os
import xml.etree.ElementTree as ET
from abc import ABC
from pathlib import Path
from typing import List, Tuple

import gym
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel

from environment.sidechannels.creation_sc import CreationSC
from environment.sidechannels.goal_sc import GoalSC
from environment.sidechannels.wall_sc import WallSC
from environment.sidechannels.workspace_sc import WorkspaceSC
from util.config import get_config


class SimEnv(gym.Env, ABC):
    """! Custom Environment that follows gym interface
    """
    metadata = {'render.modes': ['human']}
    MAX_N_MODULES = 10
    JOINT_ANGLE_STEP = 10

    def __init__(self, env_path: str, urdf: str, use_graphics: bool, worker_id: int = 0) -> None:
        """! The SimEnv class initializer.
        @param env_path: Path of the environment executable.
        @param urdf: Instance that represents the robot urdf.
        @param use_graphics: Boolean that turns graphics on or off.
        @return  An instance of the SimEnv class.
        """
        super().__init__()

        assert Path(env_path).exists(), (
            f"Given environment file path does not exist: {env_path}\n"
            f"Make sure this points to the environment executable "
            f"that make via the instructions present in the README.md."
        )

        self.env_path = env_path
        self.urdf = urdf
        self.use_graphics = use_graphics
        self.worker_id = worker_id
        self.joint_amount = 0  # set after _initialize_unity_env

        self.creation_sc, self.goal_sc, self.workspace_sc, self.wall_sc, self.u_env = self._initialize_unity_env()
        self.behavior_name = 'ManipulatorBehavior?team=0'
        self.behavior_spec = self.u_env.behavior_specs[self.behavior_name]

    def _initialize_unity_env(self) -> Tuple[CreationSC, GoalSC, WorkspaceSC, WallSC, UnityEnvironment]:
        """! A function that creates the side channels and starts the Unity environment.
        @return The created side channel objects and the mlagents UnityEnvironment.
        """
        creation_sc = CreationSC()
        goal_sc = GoalSC()
        workspace_sc = WorkspaceSC()
        wall_sc = WallSC()
        conf_channel = EngineConfigurationChannel()

        env = UnityEnvironment(file_name=self.env_path,
                               side_channels=[conf_channel, creation_sc, goal_sc, workspace_sc,
                                              wall_sc],
                               worker_id=self.worker_id,
                               no_graphics=not self.use_graphics)
        if self.use_graphics:
            timescale = 1.
        else:
            timescale = 40.
        conf_channel.set_configuration_parameters(time_scale=timescale)
        creation_sc.send_build_command(self.urdf)

        env.reset()
        while not creation_sc.creation_done:
            pass
        self.joint_amount = creation_sc.get_joint_amount()
        return creation_sc, goal_sc, workspace_sc, wall_sc, env

    def _get_unity_observations(self) -> np.ndarray:
        """! Get the observations from the Unity environment.
        @return The observations.
        """
        decision_steps, _ = self.u_env.get_steps(self.behavior_name)
        return decision_steps.obs[0][0]

    def _set_unity_actions(self, actions: np.ndarray) -> None:
        """! Send an action to the Unity environment.
        @param actions: the actions, numbers between -1 and 1.
        """
        actions = np.pad(actions, (0, self.MAX_N_MODULES - len(actions)))
        actions = actions[None, :]
        self.u_env.set_actions(self.behavior_name, action=ActionTuple(actions))

    def set_goal(self, goal: tuple) -> None:
        """! Set the coordinates of the goal visualization.
        @param goal: The coordinates: (x, y, z).
        """
        self.goal_sc.send_goal_position(goal)

    def set_workspace(self, workspace: tuple) -> None:
        """! Set the coordinates and size of the workspace visualization.
        @param workspace: The coordinates and size: (x, y, z, sideLength).
        """
        self.workspace_sc.send_workspace(workspace)

    def build_wall(self, wall: List[List[bool]]) -> None:
        """! Build a new wall
        The first wall will be built on a certain distance from the anchor.
        Every subsequent wall will be built on a certain distance from the previous wall.

        Walls are represented by a 2D array of booleans
        True means there is a tile on that 'coordinate/index'
        and False meaning there is not
        @param wall: The new wall.
        """
        self.wall_sc.send_build_command(wall)

    def remove_walls(self) -> None:
        """! Remove all walls.
        """
        self.wall_sc.remove_walls()

    def replace_walls(self, wall: List[List[bool]]) -> None:
        """! Replace all the walls, build 1 new one
        See build_wall for how a wall is represented
        @param wall: The new wall
        """
        self.remove_walls()
        self.build_wall(wall)

    def step(self, action: np.ndarray, return_observations=True) -> np.ndarray:
        """! Do 1 step in the unity environment.
        @param action: The action for this step.
        @param return_observations: Whether to return observations after the step is taken.
        @return Observations after the step is taken.
        """
        self._set_unity_actions(action)
        self.u_env.step()

        observations = None
        if return_observations:
            observations = self._get_unity_observations()

        return observations

    def reset(self) -> np.ndarray:
        """! Reset the Unity Environment, essentially starting a new episode.
        The effect is that OnEpisodeBegin is called in Unity.
        @return Observations obtained after the reset.
        """
        self.u_env.reset()
        observations = self._get_unity_observations()
        return observations

    def pause(self, steps=200) -> None:
        """! For a certain amount of steps, take no action.
        Achieved by taking "zero" actions.
        """
        for _ in range(steps):
            actions = [0] * self.joint_amount
            _ = self.step(np.array(actions))

    def close(self) -> None:
        """! Gym interface function to close the environment.
        """
        del self.creation_sc
        del self.goal_sc
        del self.wall_sc
        del self.workspace_sc
        self.u_env.close()

    def get_current_state(self) -> np.ndarray:
        """! Get the observations from the Unity environment.
        @return The observations.
        """
        return self._get_unity_observations()


def test_environment():
    """! Start the Unity environment and test certain functions like.
    actions and the functionality of side channels.
    """
    unity_executable_path = get_config().path_to_unity_executable
    urdf_filename = get_config().path_to_robot_urdf

    urdf = ET.tostring(ET.parse(urdf_filename).getroot(), encoding='unicode')

    env = SimEnv(env_path=unity_executable_path,
                 urdf=urdf,
                 use_graphics=True)

    env.pause(100)
    env.set_workspace((0, 5, 9, 4))
    env.set_goal((1, 3.5, 5.5))

    env.pause(10000)
    env.remove_walls()
    env.pause(1000)

    env.close()
