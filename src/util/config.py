import yaml

from morphevo.directed_graph_encoding import ModuleType
from morphevo.workspace_parameters import WorkspaceParameters


class Config:
    # pylint: disable=invalid-name
    class __Config:
        def __init__(self, config_file_name: str) -> None:
            with open(config_file_name, 'r', encoding='utf8') as stream:
                config = yaml.load(stream, yaml.FullLoader)

            if 'arm' in config:
                arm = config['arm']
                self.minimum_amount_modules = arm['minimum_amount_modules']
                self.maximum_amount_modules = arm['maximum_amount_modules']
                self.length_lowerbound = arm['length_lowerbound']
                self.length_upperbound = arm['length_upperbound']
                self.parse_module_choices(arm)

            if 'coevolution' in config:
                coevolution = config['coevolution']
                self.coevolution_generations = coevolution['generations']
                self.coevolution_parents = coevolution['parents']
                self.coevolution_rl_amount = coevolution['rl_amount']
                self.coevolution_children = coevolution['children']
                self.coevolution_rl_episodes = coevolution['rl_episodes']
                self.coevolution_crossover_children = coevolution['crossover_children']

            if 'morphevo' in config:
                morphevo = config['morphevo']
                self.evolution_generations = morphevo['generations']
                self.evolution_parents = morphevo['parents']
                self.evolution_children = morphevo['children']
                self.evolution_crossover_children = morphevo['crossover_children']
                self.sample_size = morphevo['sample_size']
                self.chance_of_type_mutation = morphevo['chance_of_type_mutation'] if 'chance_of_type_mutation' in morphevo else 0.15
                self.workspace_parameters = self.parse_workspace_parameters(morphevo)

            if 'rl' in config:
                rl = config['rl']
                self.episodes = rl['episodes']
                self.steps_per_episode = rl['steps_per_episode']
                self.gamma = rl['gamma']
                self.eps_end = rl['eps_end']
                self.eps_decay = rl['eps_decay']
                self.batch_size = rl['batch_size']
                self.mem_size = rl['mem_size']
                self.eps_start = rl['eps_start']
                self.hidden_nodes = rl['hidden_nodes']

                self.goal_bal_diameter = rl['goal_bal_diameter']

        def parse_workspace_parameters(self, config) -> WorkspaceParameters:
            if ('workspace_type' in config
                    and 'workspace_cube_offset' in config
                    and 'workspace_side_length' in config):
                return WorkspaceParameters(config['workspace_type'],
                                           tuple(config['workspace_cube_offset']),
                                           config['workspace_side_length'])
            return WorkspaceParameters()

        def parse_module_choices(self, config):
            self.module_choices = []

            if 'rotate' in config['movements'] and 'tilt' in config['movements']:
                self.module_choices.append(ModuleType.TILTING_AND_ROTATING)
            if 'rotate' in config['movements']:
                self.module_choices.append(ModuleType.ROTATING)
            if 'tilt' in config['movements']:
                self.module_choices.append(ModuleType.TILTING)

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Config.instance:
            Config.instance = Config.__Config(*args, **kwargs)
        return Config.instance


def set_config(config_file_name: str):
    return Config(config_file_name)


def get_config():
    return Config()
