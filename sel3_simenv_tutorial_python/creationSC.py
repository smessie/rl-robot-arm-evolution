import uuid

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.side_channel import (SideChannel, OutgoingMessage)


class CreationSC(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("2c137891-46b7-4284-94ff-3dc14a7ab993"))

        self.creation_done = False

    def send_build_command(self, urdf: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(urdf)

        super().queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.creation_done = True
