from typing import List

import numpy as np
import wandb

from util.arm import Arm


class Logger:
    def __init__(self) -> None:
        wandb.init(project='sel3-morphevo')

    def _log_metric(self, prefix: str, name: str, generation: int, values: List[float]) -> None:
        wandb.log({f'{prefix}/{name}_mean': np.mean(values)}, step=generation)
        wandb.log({f'{prefix}/{name}_std': np.std(values)}, step=generation)

    def log(self, generation: int, arms: List[Arm]) -> None:

        # performance -> coverage, redundancy
        coverages = [arm.genome.workspace.calculate_coverage()
                     for arm in arms]
        redundancies = [arm.genome.workspace.calculate_average_redundancy()
                        for arm in arms]

        self._log_metric('performance', 'coverage', generation, coverages)
        self._log_metric('performance', 'redundancy', generation, redundancies)

        # morphologies -> module_INDEX_length / total length of morphology
        for module_nr in range(10):
            lengths = []
            for arm in arms:
                if module_nr < arm.genome.amount_of_modules:
                    lengths.append(arm.genome.module_lengths[module_nr])

            if lengths:
                self._log_metric(
                    'morphology', f'module{module_nr}_length', generation, lengths)

        total_length = [float(np.sum(arm.genome.module_lengths))
                        for arm in arms]
        self._log_metric('morphology', 'total_length',
                         generation, total_length)
