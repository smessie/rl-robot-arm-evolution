import numpy as np
import wandb
from src.rl.q_table import QTable


class Logger:
    def __init__(self):
        wandb.init(project='sel3-rl-tutorial')

    def log_episode(self, episode: int, final_state: np.ndarray,
                    goal: np.ndarray, timesteps: int, q_table: QTable):

        final_distance = np.linalg.norm(final_state[:2] - goal)

        wandb.log({
            'Episode/final_distance': final_distance,
            "Episode/timesteps": timesteps,
            "Q-Table/state_coverage": q_table.calculate_state_coverage
        }, step=episode)

    def log_test(self, episode_steps: int, final_state: np.ndarray,
                    goal: np.ndarray, goal_number: int):

        final_distance = np.linalg.norm(final_state[:2] - goal)

        wandb.log({
            'goal/actions needed to reach goal': episode_steps,
        }, step=goal)
