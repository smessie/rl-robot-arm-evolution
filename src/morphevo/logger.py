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
