##
# @file
# File that brings together all morphological evolution related function like the general evolution process and all
# related functions for mutation, crossover and selection.
#

import locale
import time
from math import sqrt
from random import randint
from typing import Callable, List, Optional
from xml.dom import minidom

import numpy as np
from ray.util import ActorPool
from tqdm import tqdm

from coevolution.arm import Arm
from morphevo.evaluator import Evaluator
from morphevo.util import alternate, generate_arms, normalize
from util.config import get_config


def run_evolution(children: Optional[List[Arm]] = None) -> List[Arm]:
    """! Evolve arms with following scheme:
        - evaluate arms
        - select best arms (selection includes best arms previous generation)
        - mutate best arms
        - restart loop with mutated arms
    @param children Optional list of children to start from. Else children will be randomly generated.
    @return List of evolved arms.
    """
    config = get_config()

    # pylint: disable=no-member
    evaluators = [
              Evaluator.remote(
                  config.path_to_unity_executable,
                  use_graphics=config.morphevo_use_graphics,
                  sample_size=config.sample_size
              )
              for _ in range(config.amount_of_cores)
        ]
    pool = ActorPool(evaluators)

    parents = []
    if not children:
        children = generate_arms(amount=config.evolution_children)

    for generation in tqdm(range(config.evolution_generations), desc='Generation'):
        children = list(pool.map_unordered(
            lambda evaluator, arm: evaluator.eval_arm.remote(arm), children))

        population = children + parents

        parents = selection_fitness(population)

        save_best_genome(parents[-1], generation)

        children = mutate_with_crossover(parents)

    save_best_genome(parents[-1], config.evolution_generations)

    return parents


def selection(selection_function: Callable, population: List[Arm]) -> List[Arm]:
    """! Do a selection on a list of parents given a selection function.
    @param selection_function The function used for selection.
    @param population The parents that will be selected from.
    @return Selected parents.
    """
    return selection_function(population)


def mutate(mutation_function: Callable, parents: List[Arm]) -> List[Arm]:
    """! Do a mutation on a list of parents given a mutation function.
    @param mutation_function The function used for mutation.
    @param parents The parents that will be mutated.
    @return Mutated parents.
    """
    return mutation_function(parents)


def selection_fitness(population: List[Arm]) -> List[Arm]:
    """! Do selection on the fitness of the arm.
    @param population The population on which you want to do selection.
    @return List of selected parents.
    """
    population_fitnesses = [calculate_fitness(arm) for arm in population]

    parent_indices = np.argsort(population_fitnesses)[-get_config().evolution_parents:]
    parents = [population[i] for i in parent_indices]

    return parents


def selection_succes_rate(population: List[Arm]) -> List[Arm]:
    """! Do selection on the success_rate of the arm. This is for the coevolution loop, success_rate is
    calculated in the rl step.
    @param population The population on which you want to do selection.
    @return List of selected parents.
    """
    population_fitnesses = [arm.success_rate for arm in population]

    parent_indices = np.argsort(population_fitnesses)[-get_config().coevolution_parents:]
    parents = [population[i] for i in parent_indices]

    return parents


def selection_fitness_diversity(population: List[Arm]) -> List[Arm]:
    """! Do selection in a fitness-diversity way.
    @param population The population on which you want to do selection.
    @return List of selected parents.
    """
    current_parents = []
    for _ in range(get_config().evolution_parents):
        next_parent = select_next_parent(population, current_parents)
        population.remove(next_parent)
        current_parents.append(next_parent)

    return current_parents


def select_next_parent(population: List[Arm], parents: List[Arm]) -> Arm:
    """! Select the next parent that will be added to parents. Do this by making a consideration between
    fitness and diversity.
    @param population Current population.
    @param parents Current list of parents.
    @return The parent that scores the highes on fitness-diversity compared to already selected parents.
    """
    population_fitnesses = [calculate_fitness(arm) for arm in population]
    population_diversities = [calculate_diversity(arm, parents) for arm in population]

    selection_scores = calculate_selection_scores(population_fitnesses, population_diversities)

    next_parent_index = np.argsort(selection_scores)[0]
    next_parent = population[next_parent_index]

    return next_parent


def calculate_fitness(arm: Arm) -> float:
    """! Calculate the fitness of a genome by calculating its coverage.
    @param arm The arm of which you want to calculate fitness.
    """
    return arm.genome.workspace.calculate_coverage()


def calculate_diversity(arm: Arm, others: List[Arm]) -> float:
    """! calculate how diverse an arm is compared to others. (others is probably the already selected parents in the
    fitness-diversity selection).
    @param arm The arm you want to compare to others to calculate diversity score.
    @param others The other arms you want to compare to.
    @return Average diversity compared to others.
    """
    if not others:
        return 0
    diversity_from_others = [arm.genome.calculate_diversity_from(other.genome) for other in others]
    average_diversity = sum(diversity_from_others) / len(diversity_from_others)

    return average_diversity


def calculate_selection_scores(population_fitnesses: List[float], population_diversities: List[float]) -> List[float]:
    """! Calculate a score on which selection can be done for fitness-diversity selection. First normalize fitness and
    diversity scores. Than calculate the distance to (1,1). If fitness and diversity are 1,1 this means 100% fit and
    100% diverse from all previously selected parents.
    @param population_fitnesses The fitnesses of a population.
    @param population_diversities The diversities of a population.
    @return List of scores that combined fitness and diversity.
    """
    fitnesses_normalized = normalize(population_fitnesses)
    diversities_normalized = normalize(population_diversities)

    selection_scores = [
        sqrt((1 - fitness) ** 2 + (1 - diversity) ** 2)
        for fitness, diversity in zip(fitnesses_normalized, diversities_normalized)
    ]

    return selection_scores


def mutate_with_crossover(parents: List[Arm]) -> List[Arm]:
    """! Do mutation with morphevo configuration parameters. First make normal children by mutating parents, after that
    crossover children are added if specified in config.
    @param parents The parents of which you want to make children.
    @return List of mutated children.
    """
    config = get_config()

    children = [
        Arm(parent.genome)
        for parent in alternate(what=parents, times=config.evolution_children - config.evolution_crossover_children)
    ]
    children += create_crossover_children(parents, config.evolution_crossover_children)

    return children


def mutate_with_crossover_coevolution(parents: List[Arm]) -> List[Arm]:
    """! Do mutation with coevolution configuration parameters.
        First make normal children by mutating parents, after that
    crossover children are added if specified in config.
    @param parents The parents of which you want to make children.
    @return List of mutated children.
    """
    config = get_config()

    children = [
        Arm(parent.genome)
        for parent in alternate(what=parents, times=config.coevolution_children - config.coevolution_crossover_children)
    ]
    children += create_crossover_children(parents, config.coevolution_crossover_children)

    return children


def create_crossover_children(parents: List[Arm], amount: int):
    """! Function to create crossover children, pick 2 random parents, check if they are
    the same. If not do crossover else pick again.
    @param parents A list of parents to choose from.
    @param amount The amount of crossover children you want.
    @return List of children resulting from crossover.
    """
    if amount == 0:
        return []
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
    """! Save the genome of an arm to a file.
    @param arm The arm of which you want to save the genome.
    @param generation The generation in which the genome was evaluated.
    """
    filename = (f'output/{int(time.time())}-mu_{get_config().evolution_parents}' +
                f'-lambda_{get_config().evolution_children}-generation_{generation}.xml')

    xml_str = minidom.parseString(arm.urdf).toprettyxml(indent="    ")
    with open(filename, "w", encoding=locale.getpreferredencoding(False)) as f:
        f.write(xml_str)
    arm.urdf_path = filename
