from __future__ import annotations

import copy
import random
import time
from collections import namedtuple
from enum import Enum
from itertools import zip_longest
from typing import List, Optional

import numpy as np

from configs.env import MODULES_MAY_ROTATE, MODULES_MAY_TILT
from morphevo.urdf_generator import URDFGenerator
from morphevo.workspace import Workspace
from util.config import get_config


class Genome:
    LENGTH_LOWER_BOUND = 1
    LENGTH_UPPER_BOUND = 4
    MIN_AMOUNT_OF_MODULES = 2
    MAX_AMOUNT_OF_MODULES = 4

    def __init__(self, parent_genome: Optional[Genome] = None) -> None:
        self.module_choices = []
        if MODULES_MAY_ROTATE:
            self.module_choices.append(ModuleType.ROTATING)
            if MODULES_MAY_TILT:
                self.module_choices.append(ModuleType.TILTING_AND_ROTATING)
        if MODULES_MAY_TILT:
            self.module_choices.append(ModuleType.TILTING)

        if parent_genome is not None:
            self.anchor_can_rotate = parent_genome.anchor_can_rotate
            self.amount_of_modules = parent_genome.amount_of_modules
            self.genotype_graph = copy.deepcopy(parent_genome.genotype_graph)
            self.mutate()
        else:
            self.anchor_can_rotate = True
            self.amount_of_modules = random.randint(self.MIN_AMOUNT_OF_MODULES, self.MAX_AMOUNT_OF_MODULES)
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
                node.module_type = np.random.choice(self.module_choices)
            node = node.next

    def get_urdf(self) -> str:
        urdf_generator = URDFGenerator(self.genome_id)
        urdf_generator.add_anchor(length=1, can_rotate=self.anchor_can_rotate)
        node = self.genotype_graph.anchor.next
        while node is not None:
            for length in node.lengths:
                urdf_generator.add_module(length,
                                          can_tilt=node.module_type in (ModuleType.TILTING,
                                                                        ModuleType.TILTING_AND_ROTATING),
                                          can_rotate=node.module_type in (ModuleType.ROTATING,
                                                                          ModuleType.TILTING_AND_ROTATING))
            node = node.next
        return urdf_generator.get_urdf()

    def calculate_diversity_from(self, other_genome: Genome):
        diversity = 0
        amount_of_modules = max(self.amount_of_modules, other_genome.amount_of_modules)

        for own_node, other_node in zip_longest(self.genotype_graph, other_genome.genotype_graph):
            if not own_node or not other_node:
                diversity += 1/amount_of_modules
            elif own_node.module_type != other_node.module_type:
                diversity += 1/amount_of_modules
            else:
                length_longest_module = max(own_node.length, other_node.length)
                diversity += (abs(own_node.length - other_node.length)/length_longest_module)/amount_of_modules

        return diversity 

    def crossover(self, other_genome: Genome) -> Genome:
        genome = Genome()

        genotype_graph = Graph()
        for own_module, other_module in zip_longest(self.genotype_graph, other_genome.genotype_graph):
            if random.randint(0, 1):
                module = own_module
            else:
                module = other_module

            if module:
                genotype_graph.add_module(module.module_type, module.length)

        genome.genotype_graph = genotype_graph
        genome.amount_of_modules = len(genotype_graph)
        return genome

    def get_amount_of_joints(self):
        joints_amount = 0
        node = self.genotype_graph.anchor.next
        while node is not None:
            if node.module_type == ModuleType.TILTING_AND_ROTATING:
                joints_amount += 2 * len(node.lengths)
            else:
                joints_amount += len(node.lengths)
            node = node.next
        if self.anchor_can_rotate:
            joints_amount += 1
        return joints_amount

    def _generate_genotype_graph(self):
        genotype_graph = Graph(Node(ModuleType.ANCHOR, [1]))
        last_node = genotype_graph.anchor
        for _ in range(self.amount_of_modules):
            module_type = np.random.choice(self.module_choices)
            length = np.random.rand() * (self.LENGTH_UPPER_BOUND - self.LENGTH_LOWER_BOUND) + self.LENGTH_LOWER_BOUND
            if last_node.module_type == module_type:
                last_node.lengths.append(length)
            else:
                new_node = Node(module_type, [length])
                last_node.next = new_node
                last_node = new_node
        return genotype_graph

    def __hash__(self):
        return hash((
            self.anchor_can_rotate,
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
    def __init__(self, module_type: ModuleType, lengths: List[int]):
        self.module_type = module_type
        self.lengths = lengths
        self.next = None

    def __hash__(self):
        return hash((self.module_type, tuple(self.lengths)))


class Graph:
    def __init__(self, anchor: Optional[Node] = None):
        self.anchor = anchor

    def add_module(self, module_type: ModuleType, length: int):
        last_module = self.get_last_module()

        if last_module.module_type == module_type:
            last_module.lengths.append(length)
        else:
            new_module = Node(module_type, [length])
            last_module.next = new_module

    def get_last_module(self):
        if not self.anchor.next:
            return self.anchor

        current_module = self.anchor
        while current_module.next:
            current_module = current_module.next
        return current_module

    def __iter__(self):
        NodeTuple = namedtuple('NodeTuple', 'module_type length') 

        current_node = self.anchor
        current_node_index = 0
        while current_node:
            yield NodeTuple(current_node.module_type, current_node.lengths[current_node_index])

            current_node_index += 1
            if current_node_index >= len(current_node.lengths):
                current_node = current_node.next
                current_node_index = 0

    def __len__(self):
        return len([None for _ in self])

    def __hash__(self):
        nodes = []
        node = self.anchor
        while node is not None:
            nodes.append(node)
            node = node.next

        return hash(tuple(nodes)) 
