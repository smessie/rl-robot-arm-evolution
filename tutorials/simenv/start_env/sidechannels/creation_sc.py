import uuid

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.side_channel import (OutgoingMessage,
                                                     SideChannel)


# Create the StringLogChannel class
class CreationSC(SideChannel):

    def __init__(self) -> None:
        # Make sure this is the same UUID as in unity!
        super().__init__(uuid.UUID("2c137891-46b7-4284-94ff-3dc14a7ab993"))
        self.creation_done = False

    def send_build_command(self, urdf: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(urdf)

        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def on_message_received(self, _: IncomingMessage) -> None:
        self.creation_done = True
