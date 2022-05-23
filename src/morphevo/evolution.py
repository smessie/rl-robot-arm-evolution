import locale
import time
from math import sqrt
from random import randint
from typing import Callable, List, Optional
from xml.dom import minidom

import numpy as np
from ray.util import ActorPool
from tqdm import tqdm

from configs.env import (MORPHEVO_USE_GRAPHICS, NUM_CORES,
                         PATH_TO_UNITY_EXECUTABLE)
from morphevo.evaluator import Evaluator
from util.arm import Arm
from util.config import get_config
from util.util import alternate, generate_arms, normalize


def evolution(children: Optional[List[Arm]] = None) -> List[Arm]:
    config = get_config()

    # pylint: disable=no-member
    evaluators = [Evaluator.remote(PATH_TO_UNITY_EXECUTABLE, use_graphics=MORPHEVO_USE_GRAPHICS,
                                   sample_size=config.sample_size)
                  for _ in range(NUM_CORES)]
    pool = ActorPool(evaluators)

    parents = []
    if not children:
        children = generate_arms(amount=config.evolution_children)

    for generation in tqdm(range(config.evolution_generations), desc='Generation'):
        # Evaluate children
        children = list(pool.map_unordered(
            lambda evaluator, arm: evaluator.eval_arm.remote(arm, config), children))

        population = children + parents

        parents = selection_fitness(population)

        save_best_genome(parents[-1], generation)

        children = mutate_with_crossover(parents)

    save_best_genome(parents[-1], config.evolution_generations)

    return parents


# TODOo try to make selection something with a calculate_fitness function (maybe difficult bcs select_fit_div function)
def selection(selection_function: Callable, population: List[Arm]) -> List[Arm]:
    return selection_function(population)


def mutate(mutation_function: Callable, parents: List[Arm]) -> List[Arm]:
    return mutation_function(parents)


def selection_fitness(population: List[Arm]) -> List[Arm]:
    population_fitnesses = [calculate_fitness(arm) for arm in population]

    parent_indices = np.argsort(population_fitnesses)[-get_config().evolution_parents:]
    parents = [population[i] for i in parent_indices]

    return parents


def selection_fitness_diversity(population: List[Arm]) -> List[Arm]:
    current_parents = []
    for _ in range(get_config().evolution_parents):
        next_parent = select_next_parent(population, current_parents)
        population.remove(next_parent)
        current_parents.append(next_parent)

    return current_parents


def selection_succes_rate(population: List[Arm]) -> List[Arm]:
    population_fitnesses = [arm.success_rate for arm in population]

    parent_indices = np.argsort(population_fitnesses)[-get_config().coevolution_parents:]
    parents = [population[i] for i in parent_indices]

    return parents


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


def mutate_with_crossover(parents: List[Arm]) -> List[Arm]:
    config = get_config()

    children = [
        Arm(parent.genome)
        for parent in alternate(what=parents, times=config.evolution_children - config.evolution_crossover_children)
    ]
    children += create_crossover_children(parents, config.evolution_crossover_children)

    return children

def mutate_with_crossover_coevolution(parents: List[Arm]) -> List[Arm]:
    config = get_config()

    children = [
        Arm(parent.genome)
        for parent in alternate(what=parents, times=config.coevolution_children - config.coevolution_crossover_children)
    ]
    children += create_crossover_children(parents, config.coevolution_crossover_children)

    return children


def create_crossover_children(parents: List[Arm], amount: int):
    if len(parents) <= 1:
        raise ValueError(f"You can't do crossover on {len(parents)} parent")

    children = []
    while len(children) <= amount:
        parent1 = parents[randint(0, len(parents) - 1)]
        parent2 = parents[randint(0, len(parents) - 1)]
        while parent1 is parent2:
            parent2 = parents[randint(0, len(parents) - 1)]

        children.append(Arm(parent1.genome.crossover(parent2.genome)))

    return children


def save_best_genome(arm: Arm, generation: int):
    filename = (f'output/{int(time.time())}-mu_{get_config().evolution_parents}' +
                f'-lambda_{get_config().evolution_children}-generation_{generation}.xml')

    xml_str = minidom.parseString(arm.urdf).toprettyxml(indent="    ")
    with open(filename, "w", encoding=locale.getpreferredencoding(False)) as f:
        f.write(xml_str)
    arm.urdf_path = filename
