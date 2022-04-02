import locale
import time
from inspect import Parameter
from itertools import count
from xml.dom import minidom

import numpy as np
from env import NUM_CORES, PATH_TO_UNITY_EXECUTABLE, USE_GRAPHICS
from morphevo.evaluator import Evaluator
from morphevo.genetic_encoding import Genome
from morphevo.logger import Logger
from ray.util import ActorPool
from tqdm import tqdm


def evolution(evolution_parameters: Parameter):
    genome_indexer = count(0)

    evaluators = [Evaluator.remote(PATH_TO_UNITY_EXECUTABLE, use_graphics=USE_GRAPHICS)
                  for _ in range(NUM_CORES)]
    pool = ActorPool(evaluators)

    logger = Logger()

    parents, parent_fitnesses = [], []
    children = [Genome(next(genome_indexer)) for _ in range(evolution_parameters.LAMBDA)]

    for generation in tqdm(range(evolution_parameters.generations), desc='Generation'):
        # Evaluate children
        children = list(pool.map_unordered(
            lambda evaluator, genome: evaluator.eval_genome.remote(genome), children))

        children_fitnesses = [calculate_fitness(child) for child in children]

        # Combine children and parents in one population
        population = children + parents
        population_fitnesses = children_fitnesses + parent_fitnesses

        # Selection
        parent_indices = np.argsort(population_fitnesses)[-evolution_parameters.MU:]
        parents = [population[i] for i in parent_indices]
        parent_fitnesses = [population_fitnesses[i] for i in parent_indices]

        # Save URDF of the best genome to file
        best_genome = population[parent_indices[-1]]
        save_best_genome(best_genome, generation, evolution_parameters)

        # create new children from selected parents
        children = [
            Genome(next(genome_indexer), parent) 
            for parent in alternate_for(what=parents, times=evolution_parameters.LAMBDA)
        ]

        logger.log(generation, parents)

def calculate_fitness(genome: Genome) -> float:
    return genome.workspace.calculate_coverage()

def save_best_genome(best_genome, generation, evolution_parameters):
    filename = (f'output/{int(time.time())}-mu_{evolution_parameters.MU}' +
        f'-lambda_{evolution_parameters.LAMBDA}-generation_{generation}.xml')

    xml_str = minidom.parseString(best_genome.get_urdf()).toprettyxml(indent="    ")
    with open(filename, "w", encoding=locale.getpreferredencoding(False)) as f:
        f.write(xml_str)

def alternate_times(what, times):
    alternations = []
    for alternation, _ in zip(alternate(what), range(times)):
        alternations.append(alternation)

    return alternations

def alternate(list):
    current_index = 0
    while True:
        yield list[current_index]
        current_index = (current_index + 1) % len(list)
