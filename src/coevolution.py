import locale
import time
from xml.dom import minidom

from morphevo.evolution import (evolution, mutate, mutate_with_crossover,
                                selection, selection_succes_rate)
from rl.deep_q_learning import train
from util.arm import Arm
from util.config import get_config
from util.util import generate_arms


def start_coevolution():

    config = get_config()
    parents = []
    children = generate_arms(amount=config.coevolution_children)

    for i in range(config.coevolution_generations):
        # disable for test
        # evolved_arms = evolution(children)

        trained_arms = train(children, num_episodes=config.coevolution_rl_episodes)

        parents = selection(selection_succes_rate, trained_arms + parents)
        
        f = open("success_rate.txt", "a")
        f.write(f"generation: {i}:\n")
        for parent in parents:
            f.write(f"{parent.success_rate}\n")
        f.close()

        save_best_genome(parents[-1], f'coevolution_{i}')

        # mutate 8 parents to get 32 new children
        children = mutate(mutate_with_crossover, parents)

    save_best_genome(parents[-1], 'final_rl_best')

# ZET ERGENSANDERS DIT PAST HIER NIET BV ZET IN utils
def save_best_genome(arm: Arm, label: str):
    filename = (f'output/{int(time.time())}-mu_{get_config().evolution_parents}' +
                f'-lambda_{get_config().evolution_children}-gamma_{get_config().gamma}-{label}.xml')

    xml_str = minidom.parseString(arm.urdf).toprettyxml(indent="    ")
    with open(filename, "w", encoding=locale.getpreferredencoding(False)) as f:
        f.write(xml_str)
