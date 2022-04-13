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

    def __init__(self, genome_id: int, parent_genome: Optional[Genome] = None) -> None:
        self.genome_id = genome_id

        if parent_genome is not None:
            self.amount_of_modules = parent_genome.amount_of_modules
            self.module_lenghts = parent_genome.module_lenghts.copy()
            self.mutate()
        else:
            self.amount_of_modules = random.randint(self.MIN_AMOUNT_OF_MODULES, self.MAX_AMOUNT_OF_MODULES)
            self.module_lenghts = np.random.rand(
                self.amount_of_modules) * (self.LENGTH_UPPER_BOUND - self.LENGTH_LOWER_BOUND) + self.LENGTH_LOWER_BOUND

        self.workspace = Workspace()

    def mutate(self) -> None:
        mu, sigma = 0, 0.1
        self.module_lenghts += np.random.normal(mu, sigma, self.amount_of_modules)

        self.module_lenghts = np.clip(
            self.module_lenghts, self.LENGTH_LOWER_BOUND, self.LENGTH_UPPER_BOUND)

    def get_urdf(self) -> str:
        urdf_generator = URDFGenerator(self.genome_id)
        urdf_generator.add_anchor(length=1)
        for module_lenght in self.module_lenghts:
            urdf_generator.add_module(module_lenght)
        return urdf_generator.get_urdf()
