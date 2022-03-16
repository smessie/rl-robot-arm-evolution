from typing import List
from morph_evo.genetic_encoding import Genome
import wandb
import numpy as np


class Logger:
    def __init__(self) -> None:
        wandb.init(project='sel3-morphevo')

    def _log_metric(self, prefix: str, name: str, generation: int, values: List[float]) -> None:
        wandb.log({f'{prefix}/{name}_mean': np.mean(values)}, step=generation)
        wandb.log({f'{prefix}/{name}_std': np.std(values)}, step=generation)


    def log(self, generation: int, genomes: List[Genome]) -> None:
        # performance -> coverage, redundancy
        coverages = [genome.workspace.calculate_coverage() for genome in genomes]
        redundancies = [genome.workspace.calculate_average_redundancy() for genome in genomes]

        self._log_metric('performance', 'coverage', generation, coverages)
        self._log_metric('performance', 'redundancy', generation, redundancies)

        # morphologies -> module_INDEX_length / total length of morphology
        for i in range(4):
            lengths = [genome.module_lenghts[i] for genome in genomes]
            self._log_metric('morphology', f'module{i}_length', generation, lengths)

        total_length = [float(np.sum(genome.module_lenghts)) for genome in genomes]
        self._log_metric('morphology', 'total_length', generation, total_length)





