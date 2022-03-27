import logging
import sys
from os.path import exists

import ray

from morphevo.evolution import evolution
from morphevo.parameters import Parameters

if __name__ == '__main__':
    if not exists(sys.argv[1]):
        print(f"Configfile '{sys.argv[1]}' does not exist.")
        sys.exit()
    evolution_parameters = Parameters(sys.argv[1])
    ray.init(log_to_driver=False, logging_level=logging.WARNING)
    evolution(evolution_parameters)
