from itertools import count

import numpy as np
import psutil
from morphevo.evaluator import Evaluator
from morphevo.genetic_encoding import Genome
from morphevo.logger import Logger
from ray.util import ActorPool
from tqdm import tqdm

GENERATIONS = 100
ENV_PATH = 'environment/unity_environment/simenv.x86_64'
# MU = # parents
# LAMBDA = # children
MU, LAMBDA = 5, 5
NUM_CORES = psutil.cpu_count()


def calculate_fitness(genome: Genome) -> float:
    return genome.workspace.calculate_coverage()


def evolution():
    genome_indexer = count(0)

    evaluators = [Evaluator.remote(ENV_PATH, use_graphics=True)
                  for _ in range(NUM_CORES)]
    pool = ActorPool(evaluators)

    logger = Logger()

    parents, parent_fitnesses = [], []
    children = [Genome(next(genome_indexer)) for _ in range(LAMBDA)]

    for generation in tqdm(range(GENERATIONS), desc='Generation'):
        # Evaluate children
        children = list(pool.map_unordered(
            lambda evaluator, genome: evaluator.eval_genome.remote(genome), children))

        children_fitnesses = []
        for child in children:
            children_fitnesses.append(calculate_fitness(child))

        # Combine children and parents in one population

        population = children + parents
        population_fitnesses = children_fitnesses + parent_fitnesses

        # Selection
        parent_indices = np.argsort(population_fitnesses)[-MU:]
        parents = [population[i] for i in parent_indices]
        parent_fitnesses = [population_fitnesses[i] for i in parent_indices]

        # create new children from selected parents
        children = []
        parent_index = 0
        while len(children) < LAMBDA:
            parent = parents[parent_index]
            parent_index = (parent_index + 1) % MU

            child = Genome(next(genome_indexer), parent_genome=parent)
            children.append(child)

        logger.log(generation, parents)
