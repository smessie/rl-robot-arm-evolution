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
    children = generate_arms(amount=config.evolution_children)

    for _ in range(config.coevolution_generations):
        evolved_arms = evolution(children)

        trained_arms = train(evolved_arms)

        parents = selection(selection_succes_rate, trained_arms + parents)

        # mutate 8 parents to get 32 new children
        children = mutate(mutate_with_crossover, parents)

    save_best_genome(parents[0])


def save_best_genome(arm: Arm):
    filename = (f'output/{int(time.time())}-mu_{get_config().evolution_parents}' +
                f'-lambda_{get_config().evolution_children}-gamma_{get_config().gamma}-final_rl_best.xml')

    xml_str = minidom.parseString(arm.urdf).toprettyxml(indent="    ")
    with open(filename, "w", encoding=locale.getpreferredencoding(False)) as f:
        f.write(xml_str)
