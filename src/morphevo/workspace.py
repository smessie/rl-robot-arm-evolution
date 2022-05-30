from collections import defaultdict
from typing import List

import numpy as np


class Workspace:
    """!
    A class that represents a Workspace, which is actually the goal space i.e. all coordinates where the goal can
    appear.
    """

    def __init__(self, workspace: str = 'normalized_cube', cube_offset: tuple = (0, 0, 0), side_length: float = 13,
                 discretization_step: float = 1.0):
        """!
        @param side_length: the length of each side in case the workspace is a normalized or moved cube
        @param discretization_step: step size for discretization
        @param workspace: type of workspace: normalized_cube or moved_cube
        @param cube_offset: tuple containing the offset of the moved cube
        """
        self.side_length = side_length
        self.discretization_step = discretization_step
        self.workspace = workspace
        self.cube_offset = cube_offset

        self.n_grid_cells = int(side_length // discretization_step) ** 3
        self.grid = defaultdict(set)

    def add_end_effector_position(self, end_effector_position: np.ndarray, joint_angles: np.ndarray) -> None:
        """!
        Registers a visit of the end effector at the given position in the workspaces internal grid.

        @param end_effector_position: numpy array consisting of the position [x,y,z] of the end effector
        @param joint_angles: numpy array containing all joint angles of the genome in the form [j0angel, j0x, j0y, j0z,
         ...]
        """
        if self.workspace == 'normalized_cube':
            return self._add_end_effector_position_normalized_cube(end_effector_position, joint_angles)
        if self.workspace == 'moved_cube':
            return self._add_end_effector_position_moved_cube(end_effector_position, joint_angles)
        raise WorkspaceNotFoundError

    def _add_end_effector_position_normalized_cube(self, ee_pos: np.ndarray, joint_angles: np.ndarray) -> None:
        x, y, z = ee_pos
        if abs(x) > self.side_length / 2 or abs(z) > self.side_length / 2 or y < 0 or y > self.side_length:
            return

        x_index = int(x // self.discretization_step)
        y_index = int(y // self.discretization_step)
        z_index = int(z // self.discretization_step)

        self.grid[(x_index, y_index, z_index)].add(tuple(joint_angles))

    def _add_end_effector_position_moved_cube(self, ee_pos: np.ndarray, joint_angles: np.ndarray) -> None:
        x, y, z = ee_pos
        offset_x, offset_y, offset_z = self.cube_offset
        if self._x_axis_end_effector_outside_moved_cube(x, offset_x) or \
                self._y_axis_end_effector_outside_moved_cube(y, offset_y) or \
                self._z_axis_end_effector_outside_moved_cube(z, offset_z):
            return

        x_index = int(x // self.discretization_step)
        y_index = int(y // self.discretization_step)
        z_index = int(z // self.discretization_step)

        self.grid[(x_index, y_index, z_index)].add(tuple(joint_angles))

    def _x_axis_end_effector_outside_moved_cube(self, x, offset_x):
        return x < (offset_x - self.side_length / 2) or x > (offset_x + self.side_length / 2)

    def _y_axis_end_effector_outside_moved_cube(self, y, offset_y):
        return y < offset_y or y > (offset_y + self.side_length)

    def _z_axis_end_effector_outside_moved_cube(self, z, offset_z):
        return z < (offset_z - self.side_length / 2) or z > (offset_z + self.side_length / 2)

    def calculate_coverage(self) -> float:
        """!
        Calculates the coverage by dividing the amount of visited cells by the amount of all cells in the workspace.
        @return The calculated coverage.
        """
        return len(self.grid) / self.n_grid_cells

    def calculate_average_redundancy(self) -> float:
        """!
        Calculates the average redundancy by dividing the total amount of visits by the amount of all cells in the
         workspace.
        @return The calculated average redundancy.
        """
        visit_counts = 0
        for v in self.grid.values():
            visit_counts += len(v)

        return visit_counts / self.n_grid_cells

    def get_x_range(self) -> List[float]:
        """!
        Get the range as [min, max] of the workspace on the x-axis.
        @return A list of 2 containing the minimum and maximum.
        """
        offset_x = self.cube_offset[0]
        return [offset_x - self.side_length / 2, offset_x + self.side_length / 2]

    def get_y_range(self) -> List[float]:
        """!
        Get the range as [min, max] of the workspace on the y-axis.
        @return A list of 2 containing the minimum and maximum.
        """
        offset_y = self.cube_offset[1]
        return [max(0, offset_y - self.side_length / 2), offset_y + self.side_length / 2]

    def get_z_range(self) -> List[float]:
        """!
        Get the range as [min, max] of the workspace on the z-axis.
        @return A list of 2 containing the minimum and maximum.
        """
        offset_z = self.cube_offset[2]
        return [offset_z - self.side_length / 2, offset_z + self.side_length / 2]


class WorkspaceNotFoundError(Exception):
    """!
    Exception class invoked when a workspace is tried to use that is unknown.
    """
    pass
