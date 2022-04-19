from __future__ import annotations

import random
from enum import Enum
from typing import Optional

import numpy as np

from configs.env import MODULES_MAY_ROTATE, MODULES_MAY_TILT
from morphevo.urdf_generator import URDFGenerator
from morphevo.workspace import Workspace


class Genome:
    LENGTH_LOWER_BOUND = 1
    LENGTH_UPPER_BOUND = 4
    MIN_AMOUNT_OF_MODULES = 3
    MAX_AMOUNT_OF_MODULES = 3

    def __init__(self, genome_id: int, parent_genome: Optional[Genome] = None, workspace_type: str = 'normalized_cube',
                 workspace_cube_offset: tuple = (0, 0, 0), workspace_side_length: float = 13) -> None:
        self.genome_id = genome_id

        self.module_choices = []
        if MODULES_MAY_ROTATE:
            self.module_choices.append(ModuleType.ROTATING)
            if MODULES_MAY_TILT:
                self.module_choices.append(ModuleType.TILTING_AND_ROTATING)
        if MODULES_MAY_TILT:
            self.module_choices.append(ModuleType.TILTING)

        if parent_genome is not None:
            self.amount_of_modules = parent_genome.amount_of_modules
            self.module_lengths = parent_genome.module_lengths.copy()
            self.module_types = parent_genome.module_types.copy()
            self.mutate()
        else:
            self.amount_of_modules = random.randint(self.MIN_AMOUNT_OF_MODULES, self.MAX_AMOUNT_OF_MODULES)
            self.module_lengths = np.random.rand(
                self.amount_of_modules) * (self.LENGTH_UPPER_BOUND - self.LENGTH_LOWER_BOUND) + self.LENGTH_LOWER_BOUND
            self.module_types = np.random.choice(self.module_choices, self.amount_of_modules)

        self.workspace = Workspace(side_length=workspace_side_length, workspace=workspace_type,
                                   cube_offset=workspace_cube_offset)

    def mutate(self) -> None:
        mu, sigma = 0, 0.1
        self.module_lengths += np.random.normal(mu, sigma, self.amount_of_modules)

        self.module_lengths = np.clip(
            self.module_lengths, self.LENGTH_LOWER_BOUND, self.LENGTH_UPPER_BOUND)

        chance_of_not_inheriting_type_correctly = 0.15
        self.module_types[np.random.rand(*self.module_types.shape) < chance_of_not_inheriting_type_correctly] \
            = np.random.choice(self.module_choices)

    def get_urdf(self) -> str:
        urdf_generator = URDFGenerator(self.genome_id)
        urdf_generator.add_anchor(length=1, can_rotate=True)
        for module_length, module_type in zip(self.module_lengths, self.module_types):
            urdf_generator.add_module(module_length,
                                      can_tilt=module_type in (ModuleType.TILTING, ModuleType.TILTING_AND_ROTATING),
                                      can_rotate=module_type in (ModuleType.ROTATING, ModuleType.TILTING_AND_ROTATING))
        return urdf_generator.get_urdf()

    def calculate_diversity_from(self, other_genome: Genome):
        module_length_diversity = []

        for module_number in range(max(self.amount_of_modules, other_genome.amount_of_modules)):
            if module_number < self.amount_of_modules and module_number < other_genome.amount_of_modules:
                module_length_diversity.append(abs(self.module_lengths[module_number]))
            elif module_number < self.amount_of_modules:
                module_length_diversity += self.module_lengths[module_number:]
            else:
                module_length_diversity += other_genome.module_lengths[module_number:]
        return sum(module_length_diversity) / len(module_length_diversity)

    def crossover(self, other_genome: Genome, crossover_genome_id: int) -> Genome:
        genome = Genome(crossover_genome_id)

        # make combination of the modules
        module_lengths = []
        module_types = []
        for module_length_1, module_type_1, module_length_2, module_type_2 in \
                zip(self.module_lengths, self.module_types, other_genome.module_lengths, other_genome.module_types):
            if random.randint(0, 1):
                module_lengths.append(module_length_1)
                module_types.append(module_type_1)
            else:
                module_lengths.append(module_length_2)
                module_types.append(module_type_2)

        # maybe add leftover modules of the longest arm
        if random.randint(0, 1) and len(module_lengths) < max(self.amount_of_modules, other_genome.amount_of_modules):
            if self.amount_of_modules > other_genome.amount_of_modules:
                module_lengths += self.module_lengths[len(module_lengths):self.amount_of_modules]
                module_types += self.module_types[len(module_types):self.amount_of_modules]
            else:
                module_lengths += self.module_lengths[len(module_lengths):other_genome.amount_of_modules]
                module_types += self.module_types[len(module_types):other_genome.amount_of_modules]

        genome.module_lengths = np.array(module_lengths)
        genome.module_types = np.array(module_types)
        genome.amount_of_modules = len(genome.module_lengths)
        return genome


class ModuleType(Enum):
    TILTING = 1
    ROTATING = 2
    TILTING_AND_ROTATING = 3
