from pathlib import Path
from typing import Tuple

import os
import gym
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from src.environment.sidechannels.creationSC import CreationSC


class SimEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    MAX_N_MODULES = 10
    JOINT_ANGLE_STEP = 10

    def __init__(self, env_path: str, urdf: str, use_graphics: bool, worker_id: int = 0) -> None:
        super(SimEnv, self).__init__()

        assert Path(env_path).exists(), f"Given environment file path does not exist: {env_path}\n" \
                                        f"Make sure this points to the environment executable " \
                                        f"that you can download from Ufora."

        self.env_path = env_path
        self.urdf = urdf
        self.use_graphics = use_graphics
        self.worker_id = worker_id

        self.creation_sc, self.u_env = self._initialize_unity_env()
        self.behavior_name = 'ManipulatorBehavior?team=0'
        self.behavior_spec = self.u_env.behavior_specs[self.behavior_name]

    def _initialize_unity_env(self) -> Tuple[CreationSC, UnityEnvironment]:
        creation_sc = CreationSC()
        conf_channel = EngineConfigurationChannel()

        env = UnityEnvironment(file_name=self.env_path,
                               side_channels=[conf_channel, creation_sc],
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
        return creation_sc, env

    def _get_unity_observations(self) -> np.ndarray:
        decision_steps, terminal_steps = self.u_env.get_steps(self.behavior_name)
        return decision_steps.obs[0][0]

    def _set_unity_actions(self, actions: np.ndarray) -> None:
        # Assume the user of this already mapped the actions to -1, 0 or 1
        #   We want to allow both discrete and continuous actions to stay as generic as possible
        actions = np.pad(actions, (0, self.MAX_N_MODULES - len(actions)))
        actions = actions[None, :]
        self.u_env.set_actions(self.behavior_name, action=ActionTuple(actions))

    def step(self, actions: np.ndarray) -> np.ndarray:
        self._set_unity_actions(actions)
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


if __name__ == '__main__':
    import xml.etree.ElementTree as ET

    # make absolute paths to avoid file-not-found errors
    here = os.path.dirname(os.path.abspath(__file__))
    urdf_filename = os.path.join(here, 'robot.urdf')
    env_filename = os.path.join(here, 'unity_environment/simenv.x86_64')

    PATH_TO_YOUR_UNITY_EXECUTABLE = env_filename 
    urdf = ET.tostring(ET.parse(urdf_filename).getroot(), encoding='unicode')

    env = SimEnv(env_path=PATH_TO_YOUR_UNITY_EXECUTABLE,
                 urdf=urdf,
                 use_graphics=True)

    obs = env.reset()
    for _ in range(1000):
        # obs -> MODEL > actions
        actions = np.random.rand(3)

        actions = (actions - 0.5) * 2  # Map to [-1, 1] range
        actions[actions > 0.2] = 1
        actions[actions < -0.2] = -1
        actions[np.abs(actions) < 0.2] = 0

        # We will not actuate the first joint in this tutorial
        actions[0] = 0.

        obs = env.step(actions)

    env.close()

