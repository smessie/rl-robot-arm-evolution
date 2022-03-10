

import numpy as np


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

    def calculate_state_coverage(self) -> float:
        return len(self.table) / self.n_states
