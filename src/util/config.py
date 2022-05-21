import yaml

from morphevo.workspace_parameters import WorkspaceParameters


class Config:
    # pylint: disable=invalid-name
    class __Config:
        def __init__(self, config_file_name: str) -> None:
            with open(config_file_name, 'r', encoding='utf8') as stream:
                config = yaml.load(stream, yaml.FullLoader)

            if 'coevolution' in config:
                coevolution = config['coevolution']
                self.coevolution_generations = coevolution['coevolution_generations']
                self.coevolution_parents = coevolution['coevolution_parents']
                self.coevolution_rl_amount = coevolution['coevolution_rl_amount']
                self.coevolution_children = coevolution['coevolution_children']
                self.coevolution_rl_episodes = coevolution['coevolution_rl_episodes']

            if 'morphevo' in config:
                morphevo = config['morphevo']
                self.evolution_generations = morphevo['evolution_generations']
                self.evolution_parents = morphevo['evolution_parents']
                self.evolution_children = morphevo['evolution_children']
                self.crossover_children = morphevo['crossover_children']
                self.sample_size = morphevo['sample_size']
                self.workspace_parameters = self.parse_workspace_parameters(morphevo)

            if 'rl' in config:
                rl = config['rl']
                self.gamma = rl['gamma']
                self.eps_end = rl['eps_end']
                self.eps_decay = rl['eps_decay']
                self.batch_size = rl['batch_size']
                self.mem_size = rl['mem_size']
                self.eps_start = rl['eps_start']
                self.hidden_nodes = rl['hidden_nodes']

                self.workspace_discretization = rl['workspace_discretization']
                self.goal_bal_diameter = rl['goal_bal_diameter']

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
