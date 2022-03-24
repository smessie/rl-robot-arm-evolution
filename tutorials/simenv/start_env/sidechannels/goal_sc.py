import uuid
from typing import Tuple

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.side_channel import (OutgoingMessage,
                                                     SideChannel)


class GoalSC(SideChannel):

    def __init__(self) -> None:
        # Make sure this is the same UUID as in unity!
        super().__init__(uuid.UUID("2cc47842-41f8-455d-9ff7-5925d152133a"))
        self.goal_received = False

    def send_goal_position(self, goal: Tuple[float]) -> None:
        self.goal_received = False
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_float32_list(list(goal))

        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def on_message_received(self, _: IncomingMessage) -> None:
        self.goal_received = True
