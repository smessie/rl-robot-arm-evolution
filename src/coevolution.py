from morphevo.evolution import (mutate, mutate_with_crossover_coevolution,
                                selection, selection_succes_rate)
from rl.deep_q_learning import train
from util.config import get_config
from util.util import generate_arms, save_best_genome


def start_coevolution():

    config = get_config()
    parents = []
    children = generate_arms(amount=3*config.coevolution_children)

    for i in range(config.coevolution_generations):
        # disable for test
        # evolved_arms = evolution(children)

        trained_arms = train(children)

        parents = selection(selection_succes_rate, trained_arms + parents)

        with open("success_rate.txt", "a", encoding="utf8") as f:
            f.write(f"generation: {i}:\n")
            for index, parent in enumerate(parents):
                # save genomes
                save_best_genome(parent, f'coevolution_{i}_{index}')

                # save success rates
                f.write(f"{parent.success_rate}\n")

        #save_best_genome(parents[-1], f'coevolution_{i}')

        # mutate 8 parents to get 32 new children
        children = mutate(mutate_with_crossover_coevolution, parents)

    #save_best_genome(parents[-1], 'final_rl_best')
    for index, parent in enumerate(parents):
        save_best_genome(parent, f'coevolution_{i}_{index}')
