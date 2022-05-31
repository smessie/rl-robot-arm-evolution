##
# @file
# Various run functions that are executed based on passed command line parameters.
#
import logging
import sys
from os.path import exists

import ray

from coevolution.coevolution import run_coevolution
from environment import environment
from morphevo.evolution import run_evolution
from morphevo.util import write_morphevo_benchmarks
from rl.deep_q_learning import run_reinforcement_learning
from util.config import set_config


def start_morphevo():
    """! Run only morphological evolution
    """
    ray.init(log_to_driver=False, logging_level=logging.WARNING)
    best_genome = run_evolution()[0]
    write_morphevo_benchmarks(best_genome)


def start_rl():
    """! Run only reinforcement learning
    """
    # network path may be given when testing
    network_path = ""
    if len(sys.argv) >= 4:
        network_path = sys.argv[3]
    run_reinforcement_learning(network_path)


def start_test_env():
    """! Start Unity environment and run tests.
    """
    environment.test_environment()


def start_coevolution():
    """! Run coevolution.
    """
    ray.init(log_to_driver=False, logging_level=logging.WARNING)
    run_coevolution()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify a command ('start_test_env', 'morphevo', 'rl', 'coevolution')")
        sys.exit()
    else:
        if len(sys.argv) < 3:
            print("Please specify a config file.")
            sys.exit()
        elif not exists(sys.argv[2]):
            print(f"Configfile '{sys.argv[2]}' does not exist.")
            sys.exit()
        else:
            set_config(sys.argv[2])
            if sys.argv[1] == "morphevo":
                start_morphevo()
            elif sys.argv[1] == "rl":
                start_rl()
            elif sys.argv[1] == "coevolution":
                start_coevolution()
            elif sys.argv[1] == "start_test_env":
                start_test_env()
            else:
                print("Please specify a valid command ('start_test_env', 'morphevo', 'rl', 'coevolution')")
                sys.exit()
