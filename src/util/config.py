import yaml

from morphevo.workspace_parameters import WorkspaceParameters


class Config:
    # pylint: disable=invalid-name
    class __Config:
        def __init__(self, config_file_name: str) -> None:
            with open(config_file_name, 'r', encoding='utf8') as stream:
                config = yaml.load(stream, yaml.FullLoader)

            # morphevo parameters
            self.generations = config['generations']
            self.MU = config['mu']
            # pylint: disable=invalid-name
            self.LAMBDA = config['lambda']
            self.crossover_children = config['crossover_children']
            self.workspace_parameters = self.parse_workspace_parameters(config)

            # rl parameters
            self.gamma = config['gamma']
            self.eps_end = config['eps_end']
            self.eps_decay = config['eps_decay']
            self.batch_size = config['batch_size']
            self.mem_size = config['mem_size']
            self.eps_start = config['eps_start']
            self.hidden_nodes = config['hidden_nodes']

            self.workspace_discretization = config['workspace_discretization']
            self.goal_bal_diameter = config['goal_bal_diameter']

            # coevolution parameters
            self.coevolution_generations = config['coevolution_generations']
            self.coevolution_parents = config['coevolution_parents']
            self.coevolution_rl_amount = config['coevolution_rl_amount']

        def parse_workspace_parameters(self, config) -> WorkspaceParameters:
            if ('workspace_type' in config
                and 'workspace_cube_offset' in config
                and 'workspace_side_length' in config):

                return WorkspaceParameters(config['workspace_type'],
                                           tuple(config['workspace_cube_offset']),
                                           config['workspace_side_length'])
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
