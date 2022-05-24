from statistics import mean
from typing import List

import numpy as np
import wandb


class Logger:
    def __init__(self):
        wandb.init(project='sel3-rl-demo')

    # pylint: disable-msg=too-many-arguments
    def log_episode(self, episode: int, final_state: np.ndarray,
                    goal: np.ndarray, timesteps: int, total_finished: int, episodes_finished: List[int],
                    reward: float, eps: float):

        final_distance = np.linalg.norm(final_state[:3] - goal)

        wandb.log({
            'Distance away from the goal through time': final_distance,
            "Amount of timesteps throug time": timesteps,
            "Percentage of times the goal was reached through time": total_finished/(episode+1),
            "Percentage of times the goal was reached in the last 50 episodes": mean(episodes_finished),
            "Rewards through time": reward,
            "EPS through time": eps
        }, step=episode)

        #with open('output_testing', 'a') as file:
        #   file.write(str(episode) + "," + "{:.10f}".format(total_finished/(episode+1)) + "\n")

    def log_test(self, episode_steps: int,
                    current_position, goal: np.ndarray, goal_number: int):

        final_distance = np.linalg.norm(current_position - goal)

        wandb.log({
            'goal number/actions needed to reach goal': episode_steps,
            'goal number/final distance': final_distance,
        }, step=goal_number)
