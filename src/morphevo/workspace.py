from collections import defaultdict
from typing import List, Tuple

import numpy as np


class Workspace:
    def __init__(self, side_length: float = 13, discretization_step: float = 1.0, workspace: str = 'normalized_cube',
                 cube_offset: tuple = (0, 0, 0), cube_open_side: str = 'top'):
        """
        side_length:            the length of each side in case the workspace is a normalized or moved cube
        discretization_step:    step size for discretization
        workspace:              type of workspace: normalized_cube or moved_cube
        cube_offset:            tuple containing the offset of the moved cube
        cube_open_side:         which side of the moved_cube is open: top, bottom, left, right, front or back
        """
        self.side_length = side_length
        self.discretization_step = discretization_step
        self.workspace = workspace
        self.cube_offset = cube_offset
        self.cube_open_side = cube_open_side

        self.n_grid_cells = int(side_length // discretization_step) ** 3
        self.grid = defaultdict(set)

    def add_ee_position(self, ee_pos: np.ndarray, joint_angles: np.ndarray, joint_positions: List = None) -> None:
        if self.workspace == 'normalized_cube':
            return self._add_ee_position_normalized_cube(ee_pos, joint_angles)
        if self.workspace == 'moved_cube':
            return self._add_ee_position_moved_cube(ee_pos, joint_angles, joint_positions)
        raise WorkspaceNotFoundError

    def _add_ee_position_normalized_cube(self, ee_pos: np.ndarray, joint_angles: np.ndarray) -> None:
        x, y, z = ee_pos
        if abs(x) > self.side_length / 2 or abs(z) > self.side_length / 2 or y < 0 or y > self.side_length:
            return

        x_index = int(x // self.discretization_step)
        y_index = int(y // self.discretization_step)
        z_index = int(z // self.discretization_step)

        self.grid[(x_index, y_index, z_index)].add(tuple(joint_angles))

    def _add_ee_position_moved_cube(self, ee_pos: np.ndarray, joint_angles: np.ndarray, joint_positions: List) -> None:
        x, y, z = ee_pos
        offset_x, offset_y, offset_z = self.cube_offset
        if self._x_axis_ee_outside_moved_cube(x, offset_x) or self._y_axis_ee_outside_moved_cube(y, offset_y) \
                or self._z_axis_ee_outside_moved_cube(z, offset_z):
            return

        if self._has_joints_colliding_with_sides_of_box(joint_positions):
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

    def _has_joints_colliding_with_sides_of_box(self, joint_positions: List) -> bool:
        for joint_position in joint_positions:
            if self._joint_collide_left_or_right(joint_position) or self._joint_collide_bottom_or_top(joint_position) \
                    or self._joint_collide_back_or_front(joint_position):
                return True
        return False

    def _joint_collide_left_or_right(self, joint_position: Tuple) -> bool:
        offset_x, offset_y, offset_z = self.cube_offset
        x, y, z = joint_position
        return (((x == offset_x - self.side_length / 2 and self.cube_open_side != 'left') or
                 (x == offset_x + self.side_length / 2 and self.cube_open_side != 'right')) and
                offset_y <= y <= offset_y + self.side_length and
                offset_z - self.side_length / 2 <= z <= offset_z + self.side_length / 2)

    def _joint_collide_bottom_or_top(self, joint_position: Tuple) -> bool:
        offset_x, offset_y, offset_z = self.cube_offset
        x, y, z = joint_position
        return (((y == offset_y and self.cube_open_side != 'bottom') or
                 (y == offset_y + self.side_length and self.cube_open_side != 'top')) and
                offset_x - self.side_length / 2 <= x <= offset_x + self.side_length / 2 and
                offset_z - self.side_length / 2 <= z <= offset_z + self.side_length / 2)

    def _joint_collide_back_or_front(self, joint_position: Tuple) -> bool:
        offset_x, offset_y, offset_z = self.cube_offset
        x, y, z = joint_position
        return (((z == offset_z - self.side_length / 2 and self.cube_open_side != 'back') or
                 (z == offset_z + self.side_length / 2 and self.cube_open_side != 'front')) and
                offset_x - self.side_length / 2 <= x <= offset_x + self.side_length / 2 and
                offset_y <= y <= offset_y + self.side_length)

    def calculate_coverage(self) -> float:
        return len(self.grid) / self.n_grid_cells

    def calculate_average_redundancy(self) -> float:
        visit_counts = 0
        for v in self.grid.values():
            visit_counts += len(v)

        return visit_counts / self.n_grid_cells


class WorkspaceNotFoundError(Exception):
    pass
