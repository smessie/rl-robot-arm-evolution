from __future__ import annotations

from typing import Optional

import numpy as np
from urdf_generator import URDFGenerator
from workspace import Workspace


class Genome:
    LENGTH_LOWER_BOUND = 1
    LENGTH_UPPER_BOUND = 4

    def __init__(self, genome_id: int, parent_genome: Optional[Genome] = None) -> None:
        self.genome_id = genome_id

        if parent_genome is not None:
            self.module_lenghts = parent_genome.module_lenghts.copy()
            self.mutate()
        else:
            self.module_lenghts = np.random.rand(
                4) * (self.LENGTH_UPPER_BOUND - self.LENGTH_LOWER_BOUND) + self.LENGTH_LOWER_BOUND

        self.workspace = Workspace()

    def mutate(self) -> None:
        mu, sigma = 0, 0.1
        self.module_lenghts += np.random.normal(mu, sigma, 4)

        self.module_lenghts = np.clip(
            self.module_lenghts, self.LENGTH_LOWER_BOUND, self.LENGTH_UPPER_BOUND)

    def get_urdf(self) -> str:
        urdf_generator = URDFGenerator(self.genome_id)
        for module_lenght in self.module_lenghts:
            urdf_generator.add_module(module_lenght)
        return urdf_generator.get_urdf()
