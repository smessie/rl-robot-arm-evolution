from statistics import mean
from typing import List

import numpy as np
import wandb


class Logger:
    """! This class provides logging functionality to the reinforcement part of the code.
    """
    def __init__(self):
        wandb.init(project='sel3-rl')

    # pylint: disable-msg=too-many-arguments
    def log_episode(self, episode: int, final_state: np.ndarray,
                    goal: np.ndarray, time_steps: int, total_finished: int, episodes_finished: List[bool],
                    reward: float, eps: float):
        """! Log 1 episode.
        @param episode: Episode number
        @param final_state: State at the end of the episode
        @param goal: Goal this episode
        @param time_steps: Amount of steps taken in the episode
        @param total_finished: Total amount of times during the reinforcement run the arm reached the goal.
        @param episodes_finished: boolean array with length 50. It describes the last 50 episodes.
            True = the episode finished, False = the episode did not finish.
        @param reward: The reward of the last step in the episode
        @param eps: Epsilon parameter of the DQN at this moment.
        """

        final_distance = np.linalg.norm(final_state[:3] - goal)

        wandb.log({
            "Distance away from the goal through time": final_distance,
            "Amount of time steps through time": time_steps,
            "Percentage of times the goal was reached through time": total_finished/(episode+1),
            "Percentage of times the goal was reached in the last 50 episodes": mean(episodes_finished),
            "Rewards through time": reward,
            "EPS through time": eps
        }, step=episode)
