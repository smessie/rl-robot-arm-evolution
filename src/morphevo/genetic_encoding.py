from __future__ import annotations

import random
from typing import Optional

import numpy as np

from morphevo.urdf_generator import URDFGenerator
from morphevo.workspace import Workspace


class Genome:
    LENGTH_LOWER_BOUND = 1
    LENGTH_UPPER_BOUND = 4
    MIN_AMOUNT_OF_MODULES = 2
    MAX_AMOUNT_OF_MODULES = 4

    def __init__(self, genome_id: int, parent_genome: Optional[Genome] = None, workspace_type: str = 'normalized_cube',
                 workspace_cube_offset: tuple = (0, 0, 0), workspace_side_length: float = 13) -> None:
        self.genome_id = genome_id

        if parent_genome is not None:
            self.amount_of_modules = parent_genome.amount_of_modules
            self.module_lengths = parent_genome.module_lengths.copy()
            self.mutate()
        else:
            self.amount_of_modules = random.randint(self.MIN_AMOUNT_OF_MODULES, self.MAX_AMOUNT_OF_MODULES)
            self.module_lengths = np.random.rand(
                self.amount_of_modules) * (self.LENGTH_UPPER_BOUND - self.LENGTH_LOWER_BOUND) + self.LENGTH_LOWER_BOUND

        self.workspace = Workspace(side_length=workspace_side_length, workspace=workspace_type,
                                   cube_offset=workspace_cube_offset)

    def mutate(self) -> None:
        mu, sigma = 0, 0.1
        self.module_lengths += np.random.normal(mu, sigma, self.amount_of_modules)

        self.module_lengths = np.clip(
            self.module_lengths, self.LENGTH_LOWER_BOUND, self.LENGTH_UPPER_BOUND)

    def get_urdf(self) -> str:
        urdf_generator = URDFGenerator(self.genome_id)
        urdf_generator.add_anchor(length=1)
        for module_length in self.module_lengths:
            urdf_generator.add_module(module_length)
        return urdf_generator.get_urdf()

    def calculate_diversity_from(self, other_genome: Genome):
        module_lenght_diversity = []

        for module_number in range(max(self.amount_of_modules, other_genome.amount_of_modules)):
            if module_number < self.amount_of_modules and module_number < other_genome.amount_of_modules:
                module_lenght_diversity.append(abs(self.module_lengths[module_number]))
            elif module_number < self.amount_of_modules:
                module_lenght_diversity += self.module_lengths[module_number:]
            else:
                module_lenght_diversity += other_genome.module_lengths[module_number:]
        return sum(module_lenght_diversity)/len(module_lenght_diversity)

    def crossover(self, other_genome: Genome, crossover_genome_id: int) -> Genome:
        genome = Genome(crossover_genome_id)

        # make combination of the modules
        module_lenghts = []
        for module1, module2 in zip(self.module_lengths, other_genome.module_lengths):
            if random.randint(0,1):
                module_lenghts.append(module1)
            else:
                module_lenghts.append(module2)

        # maybe add leftover modules of longest arm
        if random.randint(0,1) and len(module_lenghts) < max(self.amount_of_modules, other_genome.amount_of_modules):
            if self.amount_of_modules > other_genome.amount_of_modules:
                module_lenghts += self.module_lengths[len(module_lenghts):self.amount_of_modules]
            else:
                module_lenghts += self.module_lengths[len(module_lenghts):other_genome.amount_of_modules]

        genome.module_lengths = np.array(module_lenghts)
        genome.amount_of_modules = len(genome.module_lengths)
        return genome
