import logging
import sys
from os.path import exists

import ray

from coevolution import start_coevolution
from environment import environment
from morphevo.evolution import evolution
from rl.deep_q_learning import rl
from util.config import set_config
from util.util import write_morphevo_benchmarks


def start_morphevo():
    if len(sys.argv) < 3:
        print("Something wrong with program arguments")
        sys.exit()
    if not exists(sys.argv[2]):
        print(f"Configfile '{sys.argv[2]}' does not exist.")
        sys.exit()
    set_config(sys.argv[2])

    ray.init(log_to_driver=False, logging_level=logging.WARNING)
    best_genome = evolution()[0]
    write_morphevo_benchmarks(best_genome)


def start_rl():
    if len(sys.argv) < 3:
        print("Something wrong with program arguments")
        sys.exit()
    if not exists(sys.argv[2]):
        print(f"Configfile '{sys.argv[2]}' does not exist.")
        sys.exit()
    set_config(sys.argv[2])

    # network path may be given when testing
    network_path = ""
    if len(sys.argv) >= 4:
        network_path = sys.argv[3]
    rl(network_path)


def start_test_env():
    environment.test_environment()


def init_coevolution():
    ray.init(log_to_driver=False, logging_level=logging.WARNING)
    if len(sys.argv) < 3:
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
        init_coevolution()
    else:
        print("Please specify a valid command ('start_test_env', 'morphevo', 'rl', 'coevolution')")
        sys.exit()
