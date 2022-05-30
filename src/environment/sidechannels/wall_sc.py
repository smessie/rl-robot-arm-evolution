import json
import uuid
from typing import List

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.side_channel import (OutgoingMessage,
                                                     SideChannel)


class WallSC(SideChannel):
    """! Side channel to build and remove walls"""

    def __init__(self) -> None:
        # Make sure this is the same UUID as in unity!
        super().__init__(uuid.UUID("428c60cd-9363-4ec1-bf5e-489ec58756f1"))

    def send_build_command(self, wall: List[List[bool]]) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(json.dumps({
            "wall": [{"row": row} for row in wall]
        }))

        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def remove_walls(self) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string("remove_walls")
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage) -> None:
        msg.read_string()
