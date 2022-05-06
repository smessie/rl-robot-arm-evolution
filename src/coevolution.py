from morphevo.evolution import (evolution, mutate, mutate_with_crossover,
                                selection, selection_succes_rate)
from rl.deep_q_learning import train
from util.config import get_config
from util.util import generate_arms


def start_coevolution():
    config = get_config()
    parents = []
    children = generate_arms(amount=config.evolution_children)

    for _ in range(config.coevolution_generations):
        evolved_arms = evolution(children)

        trained_arms = train(evolved_arms)

        parents = selection(selection_succes_rate, trained_arms + parents)

        # mutate 8 parents to get 32 new children
        children = mutate(mutate_with_crossover, parents)
