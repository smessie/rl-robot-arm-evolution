##
# @file
# File that contains the main function of the coevolution process and brings together all other sub-phases.
#
from morphevo.evolution import (mutate, mutate_with_crossover_coevolution,
                                selection, selection_succes_rate)
from rl.deep_q_learning import train
from util.config import get_config
from morphevo.util import generate_arms, save_genome


def start_coevolution():
    """!
    Run the coevolution process which uses morphological evolution and reinforcement learning for this.
    Every coevolution step, all genomes' URDF are saved.
    """
    config = get_config()
    parents = []
    children = generate_arms(amount=3*config.coevolution_children)

    for i in range(config.coevolution_generations):
        trained_arms = train(children)

        parents = selection(selection_succes_rate, trained_arms + parents)

        with open("success_rate.txt", "a", encoding="utf8") as f:
            f.write(f"generation: {i}:\n")
            for index, parent in enumerate(parents):
                # save genomes
                save_genome(parent, f'coevolution_{i}_{index}')

                # save success rates
                f.write(f"{parent.success_rate}\n")

        # mutate 8 parents to get 32 new children
        children = mutate(mutate_with_crossover_coevolution, parents)

