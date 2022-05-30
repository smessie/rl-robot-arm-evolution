class WorkspaceParameters:
    """! A dataclass that represents the parameters for a workspace
    """
    def __init__(self, workspace_type: str = 'normalized_cube',
                 workspace_cube_offset: tuple = (0, 0, 0), workspace_side_length: float = 13) -> None:
        """!
        @ieben
        """
        self.workspace_type = workspace_type
        self.workspace_cube_offset = workspace_cube_offset
        self.workspace_side_length = workspace_side_length

    def __iter__(self):
        return iter((self.workspace_type, self.workspace_cube_offset, self.workspace_side_length))
