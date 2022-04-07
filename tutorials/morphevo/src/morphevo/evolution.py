import locale
import time
from inspect import Parameter
from itertools import count
from math import sqrt
from random import randint
from typing import List
from xml.dom import minidom

import numpy as np
from env import NUM_CORES, PATH_TO_UNITY_EXECUTABLE, USE_GRAPHICS
from morphevo.evaluator import Evaluator
from morphevo.genetic_encoding import Genome
from morphevo.logger import Logger
from morphevo.utils import alternate
from ray.util import ActorPool
from tqdm import tqdm


def evolution(parameters: Parameter):
    genome_indexer = count(0)

    evaluators = [Evaluator.remote(PATH_TO_UNITY_EXECUTABLE, use_graphics=USE_GRAPHICS)
                  for _ in range(NUM_CORES)]
    pool = ActorPool(evaluators)

    logger = Logger()

    parents = []
    children = [Genome(next(genome_indexer)) for _ in range(parameters.LAMBDA)]

    for generation in tqdm(range(parameters.generations), desc='Generation'):
        # Evaluate children
        children = list(pool.map_unordered(
            lambda evaluator, genome: evaluator.eval_genome.remote(genome), children))

        population = children + parents

        parents = selection_fitness(population, parameters)

        save_best_genome(parents[0], generation, parameters)

        # create new children from selected parents
        children = [
            Genome(next(genome_indexer), parent)
            for parent in alternate(what=parents, times=parameters.LAMBDA - parameters.crossover_children)
        ]
        children += create_crossover_children(parents, parameters.crossover_children, genome_indexer)

        logger.log(generation, parents)


def selection_fitness(current_population: List[Genome], evolution_parameters) -> List[Genome]:

    population_fitnesses = [calculate_fitness(genome) for genome in current_population]

    parent_indices = np.argsort(population_fitnesses)[-evolution_parameters.MU:]
    parents = [current_population[i] for i in parent_indices]

    return parents

def selection_fitness_diversity(current_population: List[Genome], evolution_parameters) -> List[Genome]:
    current_parents = []
    for _ in range(evolution_parameters.MU):
        next_parent = select_next_parent(current_population, current_parents)
        current_population.remove(next_parent)
        current_parents.append(next_parent)

    return current_parents

def select_next_parent(population: List[Genome], parents: List[Genome]) -> Genome:

    population_fitnesses = [calculate_fitness(genome) for genome in population]
    population_diversities = [calculate_diversity(genome, parents) for genome in population]

    selection_scores = calculate_selection_scores(population_fitnesses, population_diversities)

    next_parent_index = np.argsort(selection_scores)[0]
    next_parent = population[next_parent_index]


    return next_parent

def calculate_fitness(genome: Genome) -> float:
    return genome.workspace.calculate_coverage()

def calculate_diversity(genome: Genome, others: List[Genome]) -> float:
    if not others:
        return 0
    diversity_from_others = [genome.calculate_diversity_from(other) for other in others]
    average_diversity = sum(diversity_from_others)/len(diversity_from_others)

    return average_diversity

def calculate_selection_scores(population_fitnesses: List[float], population_diversities: List[float]) -> List[float]:
    max_fitness = max(population_fitnesses) if max(population_fitnesses) > 0 else 1
    max_diversity = max(population_diversities) if max(population_diversities) > 0 else 1

    selection_scores = [
        sqrt((1 - fitness/max_fitness)**2 + (1 - diversity/max_diversity)**2)
                        for fitness, diversity in zip(population_fitnesses, population_diversities)
    ]

    return selection_scores

def create_crossover_children(parents: List[Genome], amount: int, genome_indexer):
    if len(parents) < 1:
        return []
    children = []
    while len(children) <= amount:
        parent1 = parents[randint(0, len(parents) - 1)]
        parent2 = parents[randint(0, len(parents) - 1)]
        while parent1 is parent2:
            parent2 = parents[randint(0, len(parents) - 1)]

        children.append(parent1.crossover(parent2, next(genome_indexer)))
    return children

def save_best_genome(best_genome, generation, evolution_parameters):
    filename = (f'output/{int(time.time())}-mu_{evolution_parameters.MU}' +
        f'-lambda_{evolution_parameters.LAMBDA}-generation_{generation}.xml')

    xml_str = minidom.parseString(best_genome.get_urdf()).toprettyxml(indent="    ")
    with open(filename, "w", encoding=locale.getpreferredencoding(False)) as f:
        f.write(xml_str)
