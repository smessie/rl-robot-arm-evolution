import logging
import sys
import xml.etree.ElementTree as ET

import numpy as np
import ray
from env import PATH_TO_UNITY_EXECUTABLE, USE_GRAPHICS
from environment.environment import SimEnv
from morphevo.evolution import evolution

if __name__ == '__main__':
    ray.init(log_to_driver=False, logging_level=logging.WARNING)
    evolution()
    sys.exit(0)
    PATH_TO_URDF = 'environment/urdf_example.urdf'
    urdf = ET.tostring(ET.parse(PATH_TO_URDF).getroot(), encoding='unicode')

    env = SimEnv(env_path=PATH_TO_UNITY_EXECUTABLE,
                 urdf=urdf,
                 use_graphics=USE_GRAPHICS)

    obs = env.reset()
    for _ in range(1000):
        # obs -> MODEL > actions
        actions = np.random.rand(4)

        actions = (actions - 0.5) * 2  # Map to [-1, 1] range
        actions[actions > 0.2] = 1
        actions[actions < -0.2] = -1
        actions[np.abs(actions) < 0.2] = 0

        obs = env.step(actions)

    env.close()
