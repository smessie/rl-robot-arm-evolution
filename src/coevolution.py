from morphevo.evolution import evolution
from util.config import get_config
from util.util import generate_arms


def start_coevolution():
    config = get_config()
    parents = []
    children = generate_arms(amount=config.LAMBDA)

    for _ in config.coevolution_generations:
        # evolve 32 arms and return
        evolved_arms = evolution(children)

        # temporary for pylint
        return evolved_arms, parents

    #     # rl train 16 best arms
    #     rl_arms = train(evolution_arms[0:16])

    #     # evaluate rl_arms
    #     best_arms = evaluate(rl_arms)

    #     # select best 8 arms form best_arms (8) and previos_parents (8) as new parents
    #     parents = selection(best_arms + parents)

    #     # mutate 8 parents to get 32 new children
    #     children = mutate(parents)
