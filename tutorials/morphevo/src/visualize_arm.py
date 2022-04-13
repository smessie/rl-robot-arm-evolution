import sys
import numpy as np
from env import PATH_TO_UNITY_EXECUTABLE
from environment.environment import SimEnv
from mlagents_envs.exception import UnityWorkerInUseException


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file:
        urdf = file.read()

    env = SimEnv(PATH_TO_UNITY_EXECUTABLE, urdf, True)
    obs = env.reset()
    for _ in range(1000):
        # obs -> MODEL > actions
        actions = np.random.rand(4)

        actions = (actions - 0.5) * 2  # Map to [-1, 1] range
        actions[actions > 0.2] = 1
        actions[actions < -0.2] = -1
        actions[np.abs(actions) < 0.2] = 0

        obs = env.step(actions)
