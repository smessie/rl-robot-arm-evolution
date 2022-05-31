import json
import uuid
from typing import List

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.side_channel import (OutgoingMessage,
                                                     SideChannel)


class WallSC(SideChannel):
    """! Side channel to build and remove walls"""

    def __init__(self) -> None:
        """! The WallSC class initializer. """
        # Make sure this is the same UUID as in unity!
        super().__init__(uuid.UUID("428c60cd-9363-4ec1-bf5e-489ec58756f1"))

    def send_build_command(self, wall: List[List[bool]]) -> None:
        """! Send wall string to environment via side channel
        @param wall: The wall to build
        """

        msg = OutgoingMessage()
        msg.write_string(json.dumps({
            "wall": [{"row": row} for row in wall]
        }))


        super().queue_message_to_send(msg)

    def remove_walls(self) -> None:
        """! Send command to remove all walls from the environment"""

        msg = OutgoingMessage()
        msg.write_string("remove_walls")

        super().queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage) -> None:
        """! Is called when the python side receives a message from the Unity environment.
        @param msg The message that is received in plain text.
        """
        msg.read_string()
