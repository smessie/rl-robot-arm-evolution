import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

import gym
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel
from sidechannels.creation_sc import CreationSC
from sidechannels.goal_sc import GoalSC


class SimEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    MAX_N_MODULES = 10
    JOINT_ANGLE_STEP = 10

    def __init__(self, env_path: str, urdf: str, use_graphics: bool,
                 worker_id: int = 0) -> None:
        super().__init__()

        assert Path(env_path).exists(), (
            f"Given environment file path does not exist: {env_path}\n"
            f"Make sure this points to the environment executable "
            f"that you can download from Ufora."
        )

        self.env_path = env_path
        self.urdf = urdf
        self.use_graphics = use_graphics
        self.worker_id = worker_id

        self.creation_sc, self.goal_sc, self.u_env = self._initialize_unity_env()
        self.behavior_name = 'ManipulatorBehavior?team=0'
        self.behavior_spec = self.u_env.behavior_specs[self.behavior_name]

    def _initialize_unity_env(self) -> Tuple[CreationSC, GoalSC, UnityEnvironment]:
        creation_sc = CreationSC()
        goal_sc = GoalSC()
        conf_channel = EngineConfigurationChannel()

        env = UnityEnvironment(file_name=self.env_path,
                               side_channels=[conf_channel, creation_sc, goal_sc],
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
        return creation_sc, goal_sc, env

    def _get_unity_observations(self) -> np.ndarray:
        decision_steps, _ = self.u_env.get_steps(
            self.behavior_name)
        return decision_steps.obs[0][0]

    def _set_unity_actions(self, actions: np.ndarray) -> None:
        # Assume the user of this already mapped the actions to -1, 0 or 1
        #   We want to allow both discrete and continuous actions
        #    to stay as generic as possible
        actions = np.pad(actions, (0, self.MAX_N_MODULES - len(actions)))
        actions = actions[None, :]
        self.u_env.set_actions(self.behavior_name, action=ActionTuple(actions))

    def set_goal(self, goal: tuple) -> None:
        self.goal_sc.send_goal_position(goal)

    def step(self, action: np.ndarray) -> np.ndarray:
        self._set_unity_actions(action)
        self.u_env.step()

        observations = self._get_unity_observations()

        return observations

    def reset(self) -> np.ndarray:
        self.u_env.reset()
        observations = self._get_unity_observations()
        return observations

    def close(self) -> None:
        del self.creation_sc
        self.u_env.close()


def test_environment():

    # make absolute paths to avoid file-not-found errors
    here = os.path.dirname(os.path.abspath(__file__))
    urdf_filename = os.path.join(here, 'robot.urdf')
    unity_executable_path = os.path.join(here, '../simenv.x86_64')

    urdf = ET.tostring(ET.parse(urdf_filename).getroot(), encoding='unicode')

    env = SimEnv(env_path=unity_executable_path,
                 urdf=urdf,
                 use_graphics=True)

    _ = env.reset()
    env.set_goal((3.0, 2.0, 4.0))
    # (Comments not including 'anchor module')
    # Rotate second module 90 degrees
    for _ in range(9):
        actions = [0, 0, 0, 0, 1, 0, 0]
        _ = env.step(actions)
    # Rotate first module 45 degrees
    for _ in range(9):
        actions = [0, 0, 0.5, 0, 0, 0, 0]
        _ = env.step(actions)
    # Tilt first module 45 degrees
    for _ in range(9):
        actions = [0, 0.5, 0, 0, 0, 0, 0]
        _ = env.step(actions)
    # Tilt second module 45 degrees
    for _ in range(9):
        actions = [0, 0, 0, 0.5, 0, 0, 0]
        _ = env.step(actions)
    # Tilt third module 90 degrees
    for _ in range(9):
        actions = [0, 0, 0, 0, 0, 1, 0]
        _ = env.step(actions)
    # Turn around the base running
    for _ in range(360):
        actions = [0.1, 0, 0, 0, 0, 0, 0]
        _ = env.step(actions)
    # Keep simulation running
    for _ in range(2000):
        actions = [0, 0, 0, 0, 0, 0, 0]
        _ = env.step(actions)

    env.close()


if __name__ == '__main__':

    test_environment()
