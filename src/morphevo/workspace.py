from collections import defaultdict

import numpy as np


class Workspace:
    def __init__(self, side_length: float = 13, discretization_step: float = 1.0):
        self.side_length = side_length
        self.discretization_step = discretization_step

        self.n_grid_cells = int(side_length // discretization_step) ** 3
        self.grid = defaultdict(set)

    def add_ee_position(self, ee_pos: np.ndarray, joint_angles: np.ndarray) -> None:
        x, y, z = ee_pos
        if abs(x) > self.side_length / 2 or abs(z) > self.side_length / 2 or y < 0 or y > self.side_length:
            return

        x_index = int(x // self.discretization_step)
        y_index = int(y // self.discretization_step)
        z_index = int(z // self.discretization_step)

        self.grid[(x_index, y_index, z_index)].add(tuple(joint_angles))

    def calculate_coverage(self) -> float:
        return len(self.grid) / self.n_grid_cells

    def calculate_average_redundancy(self) -> float:
        visit_counts = 0
        for v in self.grid.values():
            visit_counts += len(v)

        return visit_counts / self.n_grid_cells
