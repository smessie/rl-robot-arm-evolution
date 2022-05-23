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

Module = namedtuple('Module', 'module_type length') 

class Genome:
    LENGTH_LOWER_BOUND = 1
    LENGTH_UPPER_BOUND = 4
    MIN_AMOUNT_OF_MODULES = 2
    MAX_AMOUNT_OF_MODULES = 4

    def __init__(self, parent_genome: Optional[Genome] = None) -> None:

        if parent_genome:
            self.amount_of_modules = parent_genome.amount_of_modules
            self.genotype_graph = copy.deepcopy(parent_genome.genotype_graph)
            self.mutate()
        else:
            self.genotype_graph = self._generate_genotype_graph()

        workspace_parameters = get_config().workspace_parameters
        self.workspace = Workspace(*workspace_parameters)
        self.genome_id = hash(self)

    def mutate(self) -> None:
        mu, sigma = 0, 0.1
        node: Node = self.genotype_graph.anchor.next
        while node is not None:
            node.lengths = [np.clip(length + np.random.normal(mu, sigma), self.LENGTH_LOWER_BOUND,
                                    self.LENGTH_UPPER_BOUND) for length in node.lengths]

            if np.random.rand() < get_config().chance_of_type_mutation:
                node.module_type = np.random.choice(get_config().module_choices)
            node = node.next

    def get_urdf(self) -> str:
        urdf_generator = URDFGenerator(str(self.genome_id))
        urdf_generator.add_anchor(length=self.genotype_graph.anchor.lengths[0], can_rotate=True)

        for module in self.genotype_graph:
            urdf_generator.add_module(module.length,
                                      can_tilt=module.module_type in (ModuleType.TILTING,
                                                                    ModuleType.TILTING_AND_ROTATING),
                                      can_rotate=module.module_type in (ModuleType.ROTATING,
                                                                      ModuleType.TILTING_AND_ROTATING))
        return urdf_generator.get_urdf()

    def calculate_diversity_from(self, other_genome: Genome):
        diversity = 0
        amount_of_modules = max(self.amount_of_modules, other_genome.amount_of_modules)

        for own_module, other_module in zip_longest(self.genotype_graph, other_genome.genotype_graph):
            if not own_module or not other_module:
                diversity += 1/amount_of_modules
            elif own_module.module_type != other_module.module_type:
                diversity += 1/amount_of_modules
            else:
                length_longest_module = max(own_module.length, other_module.length)
                diversity += (abs(own_module.length - other_module.length)/length_longest_module)/amount_of_modules

        return diversity 

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
        last_module = self.get_last_module()

        if last_module.module_type == module_type:
            last_module.lengths.append(length)
        else:
            new_module = Node(module_type, [length])
            last_module.next = new_module

    def get_last_module(self):
        current_module = self.anchor
        while current_module.next:
            current_module = current_module.next

        return current_module

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
