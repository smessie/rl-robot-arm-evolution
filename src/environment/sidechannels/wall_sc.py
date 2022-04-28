import json
import sys
from typing import List
import uuid

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.side_channel import (OutgoingMessage,
                                                     SideChannel)


class WallSC(SideChannel):

    def __init__(self) -> None:
        # Make sure this is the same UUID as in unity!
        super().__init__(uuid.UUID("428c60cd-9363-4ec1-bf5e-489ec58756f1"))
        self.creation_done = False

    def send_build_command(self, wall: List[List[bool]]) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(json.dumps({
            "wall": [{"row": row} for row in wall]
        }))

        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage) -> None:
        if msg.read_string() != "Wall built":
            print("FATAL ERROR: COULD NOT BUILD WALL")
            sys.exit(0)
        self.creation_done = True
