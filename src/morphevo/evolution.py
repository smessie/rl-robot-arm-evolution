import locale
import time
from math import sqrt
from random import randint
from typing import Callable, List
from xml.dom import minidom

import numpy as np
from ray.util import ActorPool
from tqdm import tqdm

from configs.env import (MORPHEVO_USE_GRAPHICS, NUM_CORES,
                         PATH_TO_UNITY_EXECUTABLE)
from morphevo.evaluator import Evaluator
from morphevo.logger import Logger
from morphevo.utils import alternate, normalize
from util.arm import Arm
from util.config import get_config


def evolution():
    parameters = get_config()

    # pylint: disable=no-member
    evaluators = [Evaluator.remote(PATH_TO_UNITY_EXECUTABLE, use_graphics=MORPHEVO_USE_GRAPHICS)
                  for _ in range(NUM_CORES)]
    pool = ActorPool(evaluators)

    logger = Logger()

    parents = []
    children = [Arm() for _ in range(parameters.LAMBDA)]

    for generation in tqdm(range(parameters.generations), desc='Generation'):
        # Evaluate children
        children = list(pool.map_unordered(
            lambda evaluator, arm: evaluator.eval_genome.remote(arm.genome), children))

        population = children + parents

        parents = selection_fitness(population)

        save_best_genome(parents[0], generation)

        # create new children from selected parents
        children = [
            Arm(parent.genome)
            for parent in alternate(what=parents, times=parameters.LAMBDA - parameters.crossover_children)
        ]
        children += create_crossover_children(parents, parameters.crossover_children)

        logger.log(generation, parents)

    return parents


def selection(selection_function: Callable, population: List[Arm]) -> List[Arm]:
    return selection_function(population)

def selection_fitness(population: List[Arm]) -> List[Arm]:
    population_fitnesses = [calculate_fitness(arm) for arm in population]

    parent_indices = np.argsort(population_fitnesses)[-get_config().MU:]
    parents = [population[i] for i in parent_indices]

    return parents

def selection_fitness_diversity(population: List[Arm]) -> List[Arm]:
    current_parents = []
    for _ in range(get_config().MU):
        next_parent = select_next_parent(population, current_parents)
        population.remove(next_parent)
        current_parents.append(next_parent)

    return current_parents


def select_next_parent(population: List[Arm], parents: List[Arm]) -> Arm:
    population_fitnesses = [calculate_fitness(arm) for arm in population]
    population_diversities = [calculate_diversity(arm, parents) for arm in population]

    selection_scores = calculate_selection_scores(population_fitnesses, population_diversities)

    next_parent_index = np.argsort(selection_scores)[0]
    next_parent = population[next_parent_index]

    return next_parent


def calculate_fitness(arm: Arm) -> float:
    return arm.genome.workspace.calculate_coverage()

def calculate_diversity(arm: Arm, others: List[Arm]) -> float:
    if not others:
        return 0
    diversity_from_others = [arm.genome.calculate_diversity_from(other.genome) for other in others]
    average_diversity = sum(diversity_from_others) / len(diversity_from_others)

    return average_diversity


def calculate_selection_scores(population_fitnesses: List[float], population_diversities: List[float]) -> List[float]:
    fitnesses_normalized = normalize(population_fitnesses)
    diversities_normalized = normalize(population_diversities)

    selection_scores = [
        sqrt((1 - fitness) ** 2 + (1 - diversity) ** 2)
        for fitness, diversity in zip(fitnesses_normalized, diversities_normalized)
    ]

    return selection_scores


def create_crossover_children(parents: List[Arm], amount: int):
    if len(parents) < 1:
        return []
    children = []
    while len(children) <= amount:
        parent1 = parents[randint(0, len(parents) - 1)]
        parent2 = parents[randint(0, len(parents) - 1)]
        while parent1 is parent2:
            parent2 = parents[randint(0, len(parents) - 1)]

        children.append(parent1.genome.crossover(parent2.genome))
    return children


def save_best_genome(arm: Arm, generation: int):
    filename = (f'output/{int(time.time())}-mu_{get_config().MU}' +
                f'-lambda_{get_config().LAMBDA}-generation_{generation}.xml')

    xml_str = minidom.parseString(arm.urdf).toprettyxml(indent="    ")
    with open(filename, "w", encoding=locale.getpreferredencoding(False)) as f:
        f.write(xml_str)
    arm.urdf_path = filename
