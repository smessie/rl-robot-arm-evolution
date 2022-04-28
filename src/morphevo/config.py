from ray._private.ray_logging import configure_log_file
import yaml

from morphevo.workspace_parameters import WorkspaceParameters


class Config:
    # pylint: disable=invalid-name
    class __Config:
        def __init__(self, config_file_name: str) -> None:
            with open(config_file_name, 'r', encoding='utf8') as stream:
                evolution_config = yaml.load(stream, yaml.FullLoader)
            self.generations = evolution_config['generations']
            self.MU = evolution_config['mu']
            # pylint: disable=invalid-name
            self.LAMBDA = evolution_config['lambda']
            self.crossover_children = evolution_config['crossover_children']
            self.workspace_parameters = self.parse_workspace_parameters(evolution_config)

        def parse_workspace_parameters(self, evolution_config) -> WorkspaceParameters:
            if ('workspace_type' in evolution_config
                and 'workspace_cube_offset' in evolution_config
                and 'workspace_side_length' in evolution_config):

                return WorkspaceParameters(evolution_config['workspace_type'],
                                           tuple(evolution_config['workspace_cube_offset']),
                                           evolution_config['workspace_side_length'])
            else:
                return WorkspaceParameters()

    instance = None
    def __new__(cls, *args, **kwargs):
        if not Config.instance:
            Config.instance = Config.__Config(*args, **kwargs)
        return Config.instance

def set_config(config_file_name: str):
    return Config(config_file_name)

def get_config():
    return Config()
