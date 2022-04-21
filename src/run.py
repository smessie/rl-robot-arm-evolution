import logging
import sys
from os.path import exists

import ray

from environment import environment
from morphevo.evolution import evolution
from morphevo.config import Config
from rl.deep_q_learning import start_rl


def start_morphevo():
    if len(sys.argv) <= 2:
        print("Something wrong with program arguments")
        sys.exit()
    if not exists(sys.argv[2]):
        print(f"Configfile '{sys.argv[2]}' does not exist.")
        sys.exit()
    evolution_parameters = Config(sys.argv[2])

    ray.init(log_to_driver=False, logging_level=logging.WARNING)
    evolution(evolution_parameters, workspace_type="moved_cube", workspace_cube_offset=(10, 0, 10),
              workspace_side_length=10)


def start_test_env():
    environment.test_environment()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify a command ('start_test_env', 'morphevo', 'rl')")
        sys.exit()
    if sys.argv[1] == "start_test_env":
        start_test_env()
    elif sys.argv[1] == "morphevo":
        start_morphevo()
    elif sys.argv[1] == "rl":
        start_rl()
    else:
        print("Please specify a valid command ('start_test_env', 'morphevo', 'rl')")
        sys.exit()
