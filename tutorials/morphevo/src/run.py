import logging
import ray
from morphevo.evolution import evolution

if __name__ == '__main__':
    ray.init(log_to_driver=False, logging_level=logging.WARNING)
    evolution()
