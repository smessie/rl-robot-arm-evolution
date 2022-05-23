from __future__ import annotations

import copy
import random
import time
from collections import namedtuple
from enum import Enum
from itertools import zip_longest
from typing import List, Optional

import numpy as np

from morphevo.urdf_generator import URDFGenerator
from morphevo.workspace import Workspace
from util.config import get_config
from util.util import run_chance

Module = namedtuple('Module', 'module_type length') 

class Genome:
    def __init__(self, parent_genome: Optional[Genome] = None) -> None:

        if parent_genome:
            self.amount_of_modules = parent_genome.amount_of_modules
            self.genotype_graph = copy.deepcopy(parent_genome.genotype_graph)
            self.mutate()
        else:
            self.genotype_graph = self._generate_genotype_graph()

        self.workspace = Workspace(*get_config().workspace_parameters)
        self.genome_id = hash(self)

    def mutate(self) -> None:
        self.genotype_graph = self.genotype_graph.mutate()
        self.amount_of_modules = len(self.genotype_graph)

    def get_urdf(self) -> str:
        urdf_generator = URDFGenerator(str(self.genome_id))
        urdf_generator.add_anchor(length=self.genotype_graph.anchor.lengths[0], can_rotate=True)

        for module in self.genotype_graph:
            urdf_generator.add_module(module.length,
                              can_tilt=module.module_type in (ModuleType.TILTING, ModuleType.TILTING_AND_ROTATING),
                              can_rotate=module.module_type in (ModuleType.ROTATING, ModuleType.TILTING_AND_ROTATING),
                           )
        return urdf_generator.get_urdf()

    def calculate_diversity_from(self, other_genome: Genome):
        diversity = 0
        amount_of_modules = max(self.amount_of_modules, other_genome.amount_of_modules)

        for own_module, other_module in zip_longest(self.genotype_graph, other_genome.genotype_graph):
            if not own_module or not other_module:
                diversity += 1
            elif own_module.module_type != other_module.module_type:
                diversity += 1
            else:
                length_longest_module = max(own_module.length, other_module.length)
                diversity += (abs(own_module.length - other_module.length)/length_longest_module)

        return diversity / amount_of_modules

    def crossover(self, other_genome: Genome) -> Genome:
        genome = Genome()

        genotype_graph = Graph()
        for own_module, other_module in zip_longest(self.genotype_graph, other_genome.genotype_graph):
            module = own_module if random.randint(0, 1) else other_module
            if module:
                genotype_graph.add_module(module.module_type, module.length)

        genome.genotype_graph = genotype_graph
        genome.amount_of_modules = len(genotype_graph)
        return genome

    def get_amount_of_joints(self):
        joint_amount = 0
        for module in self.genotype_graph.iterate_graph(ignore_anchor=False):
            joint_amount += 2 if module.module_type == ModuleType.TILTING_AND_ROTATING else 1

        return joint_amount

    def _generate_genotype_graph(self):
        config = get_config()

        amount_of_modules = random.randint(config.minimum_amount_modules, config.maximum_amount_modules)
        genotype_graph = Graph()
        for _ in range(amount_of_modules):
            module_type = np.random.choice(get_config().module_choices)
            length = np.random.rand() * (config.length_upper_bound - config.length_upper_bound) + config.length_lower_bound
            genotype_graph.add_module(module_type, length)

        self.amount_of_modules = amount_of_modules
        return genotype_graph

    def __hash__(self):
        return hash((
            self.amount_of_modules,
            self.genotype_graph,
            time.ctime(),
        ))


class ModuleType(Enum):
    ANCHOR = 0
    TILTING = 1
    ROTATING = 2
    TILTING_AND_ROTATING = 3

class Node:
    def __init__(self, module_type: ModuleType, lengths: List[float]):
        self.module_type = module_type
        self.lengths = lengths
        self.next = None

class Graph:
    def __init__(self, anchor_length: float = 1.):
        self.anchor = Node(ModuleType.ANCHOR, [anchor_length])

    def add_module(self, module_type: ModuleType, length: float):
        head_module = self.get_last_module()

        if head_module.module_type == module_type:
            head_module.lengths.append(length)
        else:
            new_module = Node(module_type, [length])
            head_module.next = new_module

    def get_last_module(self):
        current_module = self.anchor
        while current_module.next:
            current_module = current_module.next

        return current_module

    def mutate(self):
        config = get_config()

        mutated_graph = Graph(self.anchor.lengths[0])

        drop_index = self.get_change_index(chance=config.chance_module_drop)
        add_index = self.get_change_index(chance=config.chance_module_add)

        for index, module in enumerate(self):
            if drop_index == index and add_index != index:
                continue
            elif add_index == index:
                mutated_graph.add_module(*self.get_random_module())

            module_type = np.random.choice(get_config().module_choices) if run_chance(config.chance_type_mutation) else module.module_type
            length = np.clip(module.length + np.random.normal(0, config.standard_deviation_length), config.length_lower_bound, config.length_upper_bound)
            mutated_graph.add_module(module_type, length)

        return mutated_graph

    def get_change_index(self, chance: float):
        return random.randint(0, len(self)) if run_chance(chance) else None

    def get_random_module(self):
        config = get_config()
        module_type = np.random.choice(get_config().module_choices)
        length = np.random.rand() * (config.length_upper_bound - config.length_upper_bound) + config.length_lower_bound
        return Module(module_type, length)

    def __iter__(self):
        return self.iterate_graph()

    def iterate_graph(self, ignore_anchor=True):
        current_node = self.anchor.next if ignore_anchor else self.anchor
        current_module_index = 0
        while current_node:
            yield Module(current_node.module_type, current_node.lengths[current_module_index])

            current_module_index += 1
            if current_module_index >= len(current_node.lengths):
                current_node = current_node.next
                current_module_index = 0

    def __len__(self):
        return len([None for _ in self])
