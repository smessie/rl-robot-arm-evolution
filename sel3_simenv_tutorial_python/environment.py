import gym
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from creationSC import CreationSC


class SimulatorEnvironment(gym.Env):
    MAX_N_MODULES = 10
    BEHAVIOR_NAME = 'ManipulatorBehavior?team=0'

    def __init__(self, env_path: str, worker_id: int, use_graphics: bool, urdf: str):
        super(SimulatorEnvironment, self).__init__()

        self.env_path = env_path
        self.worker_id = worker_id
        self.use_graphics = use_graphics
        self.urdf = urdf

        self.u_env = self.initialize_unity_environment()

    def initialize_unity_environment(self):
        creation_sc = CreationSC()
        config_channel = EngineConfigurationChannel()

        env = UnityEnvironment(file_name=self.env_path,
                               side_channels=[config_channel, creation_sc],
                               worker_id=self.worker_id,
                               no_graphics=not self.use_graphics)
        config_channel.set_configuration_parameters(time_scale=2.0)  # increase later on to evaluate all robots faster,
        # e.g. to 20 or 40 to see results way faster
        creation_sc.send_build_command(self.urdf)

        env.reset()

        while not creation_sc.creation_done:
            pass

        return env

    def get_unity_observations(self) -> np.ndarray:
        decision_steps, terminal_steps = self.u_env.get_steps(self.BEHAVIOR_NAME)
        return decision_steps.obs[0][0]

    def set_unity_actions(self, actions: np.ndarray) -> None:
        actions = np.pad(actions, (0, self.MAX_N_MODULES - len(actions)))
        actions = actions[None, :]  # shape [1, 10]

        self.u_env.set_actions(self.BEHAVIOR_NAME, action=ActionTuple(actions))

    def step(self, actions: np.ndarray) -> np.ndarray:
        self.set_unity_actions(actions)
        self.u_env.step()
        observations = self.get_unity_observations()
        return observations

    def reset(self) -> np.ndarray:
        self.u_env.reset()
        observations = self.get_unity_observations()
        return observations

    def close(self) -> None:
        self.u_env.close()


if __name__ == '__main__':
    f = open('robot.urdf', 'r')
    urdf = str(f.read())

    simenv = SimulatorEnvironment(env_path='./unity_environment/simenv.app',
                                  worker_id=0,
                                  use_graphics=True,
                                  urdf=urdf)

    obs = simenv.reset()

    for _ in range(10000):
        actions = np.random.rand(3)
        actions = (actions - 0.5) * 2  # Map to [-1, 1] range
        actions[actions > 0.2] = 1
        actions[actions < -0.2] = -1
        actions[np.abs(actions) < 0.2] = 0

        obs = simenv.step(actions)

