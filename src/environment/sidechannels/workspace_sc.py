import uuid
from typing import Tuple

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.side_channel import (OutgoingMessage,
                                                     SideChannel)


class WorkspaceSC(SideChannel):
    """! Side channel to send the workspace"""

    def __init__(self) -> None:
        # Make sure this is the same UUID as in unity!
        super().__init__(uuid.UUID("cf5e0f06-5f91-45b7-94a7-9ffe954f8bf9"))
        self.worksace_set = False

    def send_workspace(self, workspace: Tuple[float]) -> None:
        self.worksace_set = False
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_float32_list(list(workspace))

        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def on_message_received(self, _: IncomingMessage) -> None:
        self.worksace_set = True
