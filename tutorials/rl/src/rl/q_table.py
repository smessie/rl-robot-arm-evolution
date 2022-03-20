

import numpy as np
import matplotlib.pyplot as plt


class QTable:
    def __init__(self, n_states: int, n_actions: int, alpha: float,
                 gamma: float) -> None:
        self.n_states = n_states
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma

        self.table = {}  # state -> [0, 1.2, 0, 2.]

    def update(self, state: np.ndarray, new_state: np.ndarray,
               action_index: int, reward: float) -> None:
        state = tuple(state)
        new_state = tuple(new_state)

        if state not in self.table:
            self.table[state] = np.zeros(self.n_actions)
        if new_state not in self.table:
            self.table[new_state] = np.zeros(self.n_actions)

        old_value = self.table[state][action_index]
        estimated_optimal_future_value = np.max(self.table[new_state])

        self.table[state][action_index] = (
            (1 - self.alpha) * old_value + self.alpha *
            (reward + self.gamma * estimated_optimal_future_value)
        )

    def lookup(self, state: np.ndarray) -> int:
        try:
            return int(np.argmax(self.table[tuple(state)]))
        except KeyError:
            return np.random.randint(self.n_actions)

    def visualize(self):
        self.visualize_q_values(0)
        self.visualize_q_values(1)
        self.visualize_q_values(2)
        self.visualize_q_values(3)
        self.visualize_quadrant(-1, -1)
        self.visualize_quadrant(-1, 1)
        self.visualize_quadrant(1, -1)
        self.visualize_quadrant(1, 1)

    def visualize_q_values(self, action_index):
        filtered_q_table = []
        for state, action_rewards in self.table.items():
            filtered_q_table.append((state[0], state[1], action_rewards[action_index]))

        max_y = -np.inf
        max_z = -np.inf
        for y, z, _ in filtered_q_table:
            if y > max_y:
                max_y = y
            if z > max_z:
                max_z = z
        grid = [[0 for _ in range(max_z + 1)] for _ in range(max_y + 1)]
        for y, z, q_value in filtered_q_table:
            grid[y][z] += q_value
        plt.imshow(grid)
        plt.title(f"q_values for action_index {action_index}")
        plt.gca().invert_yaxis()
        plt.set_cmap("binary")
        plt.show()

    def visualize_quadrant(self, dir_y, dir_z):

        filtered_q_table = set()
        for state, action_rewards in self.table.items():
            if (state[2] == dir_y) and (state[3] == dir_z):
                filtered_q_table.add((state[0], state[1], np.argmax(action_rewards)))

        max_y = -np.inf
        max_z = -np.inf
        for y, z, _ in filtered_q_table:
            if y > max_y:
                max_y = y
            if z > max_z:
                max_z = z
        grid = [[-1 for _ in range(max_z+1)] for _ in range(max_y+1)]
        for y, z, index in filtered_q_table:
            grid[y][z] = index
        plt.imshow(grid)
        plt.title(f"action indices for y_dir={dir_y}, z_dir={dir_z}")
        plt.gca().invert_yaxis()
        for y, z, label in filtered_q_table:
            plt.text(z, y, label, ha='center', va='center')
        plt.show()

    def calculate_state_coverage(self) -> float:
        return len(self.table) / self.n_states
