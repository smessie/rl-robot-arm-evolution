import uuid

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    OutgoingMessage,
)


# Create the StringLogChannel class
class CreationSC(SideChannel):

    def __init__(self) -> None:
        # Make sure this is the same UUID as in unity!
        super().__init__(uuid.UUID("14630201-7c67-41b5-aab7-f4d87aa496a5"))
        self.creation_done = False

    def send_build_command(self, urdf: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(urdf)

        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.creation_done = True
