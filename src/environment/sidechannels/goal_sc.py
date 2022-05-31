import uuid
from typing import Tuple

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.side_channel import (OutgoingMessage,
                                                     SideChannel)


class GoalSC(SideChannel):
    """! Side channel to send a goal to the environment."""

    def __init__(self) -> None:
        """! The GoalSC class initializer. """
        # Make sure this is the same UUID as in unity!
        super().__init__(uuid.UUID("2cc47842-41f8-455d-9ff7-5925d152133a"))
        self.goal_set = False

    def send_goal_position(self, goal: Tuple[float]) -> None:
        """! Send goal string to environment via side channel.
        @param goal: The goal to show.
        """
        self.goal_set = False

        msg = OutgoingMessage()
        msg.write_float32_list(list(goal))

        super().queue_message_to_send(msg)

    def on_message_received(self, _: IncomingMessage) -> None:
        """! Is called when the python side receives a message from the Unity environment.
        @param msg The message that is received in plain text.
        """
        self.goal_set = True
