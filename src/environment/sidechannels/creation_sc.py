import json
import sys
import uuid

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.side_channel import (OutgoingMessage,
                                                     SideChannel)


class CreationSC(SideChannel):
    """! Side channel to create the robot arm"""

    def __init__(self) -> None:
        """! The CreationSC class initializer. """
        # Make sure this is the same UUID as in unity!
        super().__init__(uuid.UUID("2c137891-46b7-4284-94ff-3dc14a7ab993"))
        self.creation_done = False
        self.info = None

    def send_build_command(self, urdf: str) -> None:
        """! Send urdf string to environment via side channel
        @param urdf: The urdf to build
        """
        msg = OutgoingMessage()
        msg.write_string(urdf)

        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage) -> None:
        """! Is called when the python side receives a message from the Unity environment.
        @param msg The message that is received in plain text.
        """
        self.info = json.loads(msg.read_string())
        if self.info['Status'] != "success":
            print("FATAL ERROR: COULD NOT BUILD ROBOT")
            sys.exit(0)
        self.creation_done = True

    def get_joint_amount(self):
        """! Get the amount of joints that are present in the robot arm. """
        return self.info['JointAmount']
