import locale

import numpy as np

from util.arm import Arm


def generate_arms(amount: int):
    return [Arm() for _ in range(amount)]


def alternate(what, times):
    alternations = []
    for alternation, _ in zip(alternate_infinite(what), range(times)):
        alternations.append(alternation)

    return alternations


def alternate_infinite(what):
    current_index = 0
    while True:
        yield what[current_index]
        current_index = (current_index + 1) % len(what)


def normalize(raw):
    sum_raw = sum(raw) if sum(raw) != 0 else 1
    return [i / sum_raw for i in raw]


def write_morphevo_benchmarks(arm):
    with open("morphevo-benchmarks.csv", 'a', encoding=locale.getpreferredencoding(False)) as file:
        module_lengths = np.array([])
        node = arm.genome.genotype_graph.anchor.next
        while node is not None:
            module_lengths = np.concatenate((module_lengths, node.lengths))
            node = node.next
        file.write(f'{arm.genome.workspace.side_length},{arm.genome.workspace.cube_offset},'
                   f'{sum(module_lengths)},{arm.genome.amount_of_modules},'
                   f'{arm.genome.workspace.calculate_coverage()}\n')
