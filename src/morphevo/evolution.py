import locale
import time
from itertools import count
from math import sqrt
from random import randint
from typing import List
from xml.dom import minidom

import numpy as np
from ray.util import ActorPool
from tqdm import tqdm

from configs.env import NUM_CORES, PATH_TO_UNITY_EXECUTABLE, USE_GRAPHICS
from morphevo.evaluator import Evaluator
from morphevo.genetic_encoding import Genome
from morphevo.logger import Logger
from morphevo.utils import alternate, normalize
from rl.deep_q_learning import DeepQLearner


def evolution(parameters, workspace_type: str = 'normalized_cube',
              workspace_cube_offset: tuple = (0, 0, 0), workspace_side_length: float = 13):
    genome_indexer = count(0)

    # pylint: disable=no-member
    evaluators = [Evaluator.remote(PATH_TO_UNITY_EXECUTABLE, use_graphics=USE_GRAPHICS)
                  for _ in range(NUM_CORES)]
    pool = ActorPool(evaluators)

    logger = Logger()

    parents = []
    children = [Genome(next(genome_indexer), workspace_type=workspace_type, workspace_cube_offset=workspace_cube_offset,
                       workspace_side_length=workspace_side_length)
                for _ in range(parameters.LAMBDA)]

    for generation in tqdm(range(parameters.generations), desc='Generation'):
        # Evaluate children
        children = list(pool.map_unordered(
            lambda evaluator, genome: evaluator.eval_genome.remote(genome), children))

        population = children + parents

        parents = selection_fitness(population, parameters, generation)

        save_best_genome(parents[0], generation, parameters)

        # create new children from selected parents
        children = [
            Genome(next(genome_indexer), parent_genome=parent, workspace_type=workspace_type,
                   workspace_cube_offset=workspace_cube_offset, workspace_side_length=workspace_side_length)
            for parent in alternate(what=parents, times=parameters.LAMBDA - parameters.crossover_children)
        ]
        children += create_crossover_children(parents, parameters.crossover_children, genome_indexer)

        logger.log(generation, parents)


def selection_fitness(current_population: List[Genome], evolution_parameters, generation=0) -> List[Genome]:
    population_fitnesses = [calculate_fitness(genome) for genome in current_population]

    parent_indices = np.argsort(population_fitnesses)[-evolution_parameters.MU:]
    parents = [current_population[i] for i in parent_indices]

    use_rl_fitness_every_x_generations = 10
    avg_coverage = np.sum([genome.workspace.calculate_coverage() for genome in current_population]) / len(
        current_population)
    if avg_coverage > 0:
        if generation % use_rl_fitness_every_x_generations == 0:
            parents = rl_selection_fitness(parents, evolution_parameters)

    return parents


def rl_selection_fitness(current_population: List[Genome], evolution_parameters) -> List[Genome]:
    population_fitnesses = [calculate_rl_fitness(genome) for genome in current_population]

    parent_indices = np.argsort(population_fitnesses)[-evolution_parameters.MU // 2:]
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


def calculate_rl_fitness(genome: Genome) -> float:
    model = DeepQLearner(env_path=PATH_TO_UNITY_EXECUTABLE, urdf=genome.get_urdf())
    return model.get_score(genome.get_amount_of_joints(), genome.workspace)


def calculate_diversity(genome: Genome, others: List[Genome]) -> float:
    if not others:
        return 0
    diversity_from_others = [genome.calculate_diversity_from(other) for other in others]
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
