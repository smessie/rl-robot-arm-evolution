import logging
import sys
from os.path import exists

import ray

from coevolution import start_coevolution
from environment import environment
from morphevo.config import get_config, set_config
from morphevo.evolution import evolution
from rl.deep_q_learning import start_rl


def start_morphevo():
    if len(sys.argv) <= 2:
        print("Something wrong with program arguments")
        sys.exit()
    if not exists(sys.argv[2]):
        print(f"Configfile '{sys.argv[2]}' does not exist.")
        sys.exit()
    set_config(sys.argv[2])

    ray.init(log_to_driver=True, logging_level=logging.WARNING)
    evolution(get_config())


def start_test_env():
    environment.test_environment()

def init_coevolution():
    ray.init(log_to_driver=False, logging_level=logging.WARNING)
    if len(sys.argv) <= 2:
        print("Something wrong with program arguments")
        sys.exit()
    if not exists(sys.argv[2]):
        print(f"Configfile '{sys.argv[2]}' does not exist.")
        sys.exit()
    set_config(sys.argv[2])
    start_coevolution()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify a command ('start_test_env', 'morphevo', 'rl', 'coevolution')")
        sys.exit()
    if sys.argv[1] == "start_test_env":
        start_test_env()
    elif sys.argv[1] == "morphevo":
        start_morphevo()
    elif sys.argv[1] == "rl":
        start_rl()
    elif sys.argv[1] == "coevolution":
        start_coevolution()
    else:
        print("Please specify a valid command ('start_test_env', 'morphevo', 'rl')")
        sys.exit()
