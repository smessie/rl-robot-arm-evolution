from typing import List

import numpy as np

import wandb
from morphevo.genetic_encoding import Genome


class Logger:
    def __init__(self) -> None:
        wandb.init(project='sel3-morphevo')

    def _log_metric(self, prefix: str, name: str, generation: int, values: List[float]) -> None:
        wandb.log({f'{prefix}/{name}_mean': np.mean(values)}, step=generation)
        wandb.log({f'{prefix}/{name}_std': np.std(values)}, step=generation)

    def log(self, generation: int, genomes: List[Genome]) -> None:

        # performance -> coverage, redundancy
        coverages = [genome.workspace.calculate_coverage()
                     for genome in genomes]
        redundancies = [genome.workspace.calculate_average_redundancy()
                        for genome in genomes]

        self._log_metric('performance', 'coverage', generation, coverages)
        self._log_metric('performance', 'redundancy', generation, redundancies)

        # morphologies -> module_INDEX_length / total length of morphology
        for module_nr in range(10):
            lengths = []
            for genome in genomes:
                if module_nr < genome.amount_of_modules:
                    lengths.append(genome.module_lenghts[module_nr])

            if lengths:
                self._log_metric(
                    'morphology', f'module{module_nr}_length', generation, lengths)

        total_length = [float(np.sum(genome.module_lenghts))
                        for genome in genomes]
        self._log_metric('morphology', 'total_length',
                         generation, total_length)
