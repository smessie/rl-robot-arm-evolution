from __future__ import annotations

import copy
import random
import time
from collections import namedtuple
from itertools import zip_longest
from typing import List, Optional

import numpy as np

from morphevo.urdf_generator import URDFGenerator
from morphevo.workspace import Workspace
from util.config import ModuleType, get_config


class Genome:
    """! A class that represents a Genome, it contains an encoding, methods for evaluation, methods to mutate and
    methods to compare to other genomes.
    """
    def __init__(self, parent_genome: Optional[Genome] = None) -> None:
        """! Create a Genome, if no parent is given a random encoding will be generated. Otherwise it will
        use the mutated encoding of its parent as encoding.
        @param parent_genome The optional parent.
        """
        if parent_genome:
            self.amount_of_modules = parent_genome.amount_of_modules
            self.genotype_graph = copy.deepcopy(parent_genome.genotype_graph)
            self.mutate()
        else:
            self.genotype_graph = self._generate_random_genotype_graph()

        self.workspace = Workspace(*get_config().workspace_parameters)
        self.genome_id = hash(self)

    def mutate(self) -> None:
        """! Mutate the encoding of the genome.
        """
        self.genotype_graph = self.genotype_graph.mutate()
        self.amount_of_modules = len(self.genotype_graph)

    def get_urdf(self) -> str:
        """! Get the urdf string for an encoding.
        @return The urdf string for an encoding.
        """
        urdf_generator = URDFGenerator(str(self.genome_id))
        urdf_generator.add_anchor(length=self.genotype_graph.anchor.lengths[0], can_rotate=True)

        for module in self.genotype_graph:
            urdf_generator.add_module(module.length,
                              can_tilt=module.module_type in (ModuleType.TILTING, ModuleType.TILTING_AND_ROTATING),
                              can_rotate=module.module_type in (ModuleType.ROTATING, ModuleType.TILTING_AND_ROTATING),
                           )
        return urdf_generator.get_urdf()

    def calculate_diversity_from(self, other_genome: Genome) -> float:
        """! Calculate the diversity to another genome. This function compares modules on the same level
        of the arm. If a module is a different type the difference for that level is 1 on the amount of
        modules in the longest arm (same for missing modules). Else you calculate the percentage of length
        difference between the modules.

        @param other_genome The genome you want to compare with.
        @return A percentage on how different the genomes are.
        """
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
        """! Do a crossover with another genome. For every module you will run a chance to determine
        of which parent you will include its module. If one parent is shorter the top modules of the other
        parent can still be included at the end of the arm.
        @param other_genome The other genome to crossover with.
        @return A new genome with an encoding that is the result of crossover on its parents.
        """
        genome = Genome()

        genotype_graph = GenotypeGraph()
        for own_module, other_module in zip_longest(self.genotype_graph, other_genome.genotype_graph):
            module = own_module if random.randint(0, 1) else other_module
            if module:
                genotype_graph.add_module(module.module_type, module.length)

        genome.genotype_graph = genotype_graph
        genome.amount_of_modules = len(genotype_graph)
        return genome

    def get_amount_of_joints(self) -> int:
        """! Count the amount of joints a genotype represents.
        @retursn amount of joints.
        """
        joint_amount = 0
        for module in self.genotype_graph.iterate_graph(ignore_anchor=False):
            joint_amount += 2 if module.module_type == ModuleType.TILTING_AND_ROTATING else 1

        return joint_amount

    def _generate_random_genotype_graph(self) -> GenotypeGraph:
        """! Generate a new random genotype encoding.
        @return A random GenotypeGraph.
        """
        config = get_config()

        amount_of_modules = random.randint(config.minimum_amount_modules, config.maximum_amount_modules)
        genotype_graph = GenotypeGraph()
        for _ in range(amount_of_modules + 1):
            module = genotype_graph.get_random_module()
            genotype_graph.add_module(*module)

        self.amount_of_modules = amount_of_modules
        return genotype_graph

    def __hash__(self) -> int:
        return hash((
            self.genotype_graph,
            time.ctime(),
        ))

#
# Namedtuple that represents a module of the arm.
#
Module = namedtuple('Module', 'module_type length')

class GenotypeGraphNode:
    """! A node in the GenotypeGraph class.
    Represents the encoding of one or more modules of the same type.
    """
    def __init__(self, module_type: ModuleType, lengths: List[float]) -> None:
        """!
        @param module_type The type of the module(s) that will be represented.
        @param lengths The length(s) of the represented module(s).
        """
        self.module_type = module_type
        self.lengths = lengths
        self.next = None

class GenotypeGraph:
    """! GenotypeGraph is a directed graph encoding for the genome of an arm.
    """
    def __init__(self, anchor_length: float = 1.0) -> None:
        """!
        @param anchor_length The length of the anchor module.
        """
        self.anchor = GenotypeGraphNode(ModuleType.ANCHOR, [anchor_length])

    def add_module(self, module_type: ModuleType, length: float) -> None:
        """! Add a module to the graph, will join nodes if type is the same as last node.
        @param module_type The type of the module.
        @param length The length of the module.
        """
        head_module = self.get_last_node()

        if head_module.module_type == module_type:
            head_module.lengths.append(length)
        else:
            new_module = GenotypeGraphNode(module_type, [length])
            head_module.next = new_module

    def get_last_node(self) -> GenotypeGraphNode:
        """! Get the last node in the graphencoding.
        @return The last node in the graphencoding.
        """
        current_module = self.anchor
        while current_module.next:
            current_module = current_module.next

        return current_module

    def mutate(self) -> GenotypeGraph:
        """! Create a new graphencoding that is a mutation of itself.
        @return The mutated graphencoding.
        """
        config = get_config()

        mutated_graph = GenotypeGraph(self.anchor.lengths[0])

        drop_index = self.get_change_index(chance=config.chance_module_drop)
        drop_index = drop_index if len(self) > config.minimum_amount_modules else None

        add_index = self.get_change_index(chance=config.chance_module_add)
        add_index = add_index if len(self) < config.maximum_amount_modules else None

        for index, module in enumerate(self):
            if drop_index == index and add_index != index:
                continue
            if add_index == index:
                mutated_graph.add_module(*self.get_random_module())

            if run_chance(config.chance_type_mutation):
                module_type = np.random.choice(get_config().module_choices)
            else:
                module_type =module.module_type
            length = np.clip(
                module.length + np.random.normal(0, config.standard_deviation_length),
                config.length_lower_bound, config.length_upper_bound
            )
            mutated_graph.add_module(module_type, length)

        return mutated_graph

    def get_change_index(self, chance: float) -> int | None:
        """!
        Get the index where a change (addition of deletion) can occure. This depends on the outcome of
        a chance. If chance fails None is returned to represent no mutation.

        @return Index of mutation or None.
        """
        return random.randint(0, len(self)) if run_chance(chance) else None

    def get_random_module(self) -> Module:
        """!
        Iterator to iterate graph module after module instead of node after node.

        @return Module
        """
        config = get_config()
        module_type = np.random.choice(get_config().module_choices)
        length = np.random.rand() * (config.length_upper_bound - config.length_lower_bound) + config.length_lower_bound
        return Module(module_type, length)

    def iterate_graph(self, ignore_anchor: bool=True):
        """!
        Iterator to iterate graph module after module instead of node after node.

        @param ignore_anchor Choice to skip anchor in iteration.
        """
        current_node = self.anchor.next if ignore_anchor else self.anchor
        current_module_index = 0
        while current_node:
            yield Module(current_node.module_type, current_node.lengths[current_module_index])

            current_module_index += 1
            if current_module_index >= len(current_node.lengths):
                current_node = current_node.next
                current_module_index = 0

    def __iter__(self):
        return self.iterate_graph()

    def __len__(self):
        """!
        Return amount of modules in the arm without anchor
        """
        return len([None for _ in self])

def run_chance(chance: float) -> bool:
    """! Run a chance.
    @param chance Chance of being True on 1
    @return True or False depending on result of chance test.
    """
    return random.uniform(0,1) < chance
