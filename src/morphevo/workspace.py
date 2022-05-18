from collections import defaultdict
from typing import List

import numpy as np


class Workspace:
    def __init__(self, workspace: str = 'normalized_cube', cube_offset: tuple = (0, 0, 0), side_length: float = 13,
                 discretization_step: float = 1.0):
        """
        side_length:            the length of each side in case the workspace is a normalized or moved cube
        discretization_step:    step size for discretization
        workspace:              type of workspace: normalized_cube or moved_cube
        cube_offset:            tuple containing the offset of the moved cube
        """
        self.side_length = side_length
        self.discretization_step = discretization_step
        self.workspace = workspace
        self.cube_offset = cube_offset

        self.n_grid_cells = int(side_length // discretization_step) ** 3
        self.grid = defaultdict(set)

    def add_ee_position(self, ee_pos: np.ndarray, joint_angles: np.ndarray) -> None:
        if self.workspace == 'normalized_cube':
            return self._add_ee_position_normalized_cube(ee_pos, joint_angles)
        if self.workspace == 'moved_cube':
            return self._add_ee_position_moved_cube(ee_pos, joint_angles)
        raise WorkspaceNotFoundError

    def _add_ee_position_normalized_cube(self, ee_pos: np.ndarray, joint_angles: np.ndarray) -> None:
        x, y, z = ee_pos
        if abs(x) > self.side_length / 2 or abs(z) > self.side_length / 2 or y < 0 or y > self.side_length:
            return

        x_index = int(x // self.discretization_step)
        y_index = int(y // self.discretization_step)
        z_index = int(z // self.discretization_step)

        self.grid[(x_index, y_index, z_index)].add(tuple(joint_angles))

    def _add_ee_position_moved_cube(self, ee_pos: np.ndarray, joint_angles: np.ndarray) -> None:
        x, y, z = ee_pos
        offset_x, offset_y, offset_z = self.cube_offset
        if self._x_axis_ee_outside_moved_cube(x, offset_x) or self._y_axis_ee_outside_moved_cube(y, offset_y) \
                or self._z_axis_ee_outside_moved_cube(z, offset_z):
            return

        x_index = int(x // self.discretization_step)
        y_index = int(y // self.discretization_step)
        z_index = int(z // self.discretization_step)

        self.grid[(x_index, y_index, z_index)].add(tuple(joint_angles))

    def _x_axis_ee_outside_moved_cube(self, x, offset_x):
        return x < (offset_x - self.side_length / 2) or x > (offset_x + self.side_length / 2)

    def _y_axis_ee_outside_moved_cube(self, y, offset_y):
        return y < offset_y or y > (offset_y + self.side_length)

    def _z_axis_ee_outside_moved_cube(self, z, offset_z):
        return z < (offset_z - self.side_length / 2) or z > (offset_z + self.side_length / 2)

    def calculate_coverage(self) -> float:
        return len(self.grid) / self.n_grid_cells

    def calculate_average_redundancy(self) -> float:
        visit_counts = 0
        for v in self.grid.values():
            visit_counts += len(v)

        return visit_counts / self.n_grid_cells

    def get_x_range(self) -> List[float]:
        offset_x = self.cube_offset[0]
        return [offset_x - self.side_length / 2, offset_x + self.side_length / 2]

    def get_y_range(self) -> List[float]:
        offset_y = self.cube_offset[1]
        return [offset_y - self.side_length / 2, offset_y + self.side_length / 2]

    def get_z_range(self) -> List[float]:
        offset_z = self.cube_offset[2]
        return [offset_z - self.side_length / 2, offset_z + self.side_length / 2]


class WorkspaceNotFoundError(Exception):
    pass
