from enum import Enum

import yaml

from morphevo.workspace_parameters import WorkspaceParameters


class ModuleType(Enum):
    ANCHOR = 0
    TILTING = 1
    ROTATING = 2
    TILTING_AND_ROTATING = 3

class Config:
    # pylint: disable=invalid-name
    class __Config:
        # pylint: disable=too-many-statements
        def __init__(self, config_file_name: str) -> None:
            with open(config_file_name, 'r', encoding='utf8') as stream:
                config = yaml.load(stream, yaml.FullLoader)

            if 'environment' in config:
                environment = config['environment']
                self.path_to_unity_executable = environment['path_to_unity_executable']
                self.path_to_robot_urdf = environment['path_to_robot_urdf']
                self.morphevo_use_graphics = environment['morphevo_use_graphics']
                self.rl_use_graphics_training = environment['rl_use_graphics_training']
                self.rl_use_graphics_testing = environment['rl_use_graphics_testing']
                self.amount_of_cores = environment['amount_of_cores']

            if 'arm' in config:
                arm = config['arm']
                self.minimum_amount_modules = arm['minimum_amount_modules']
                self.maximum_amount_modules = arm['maximum_amount_modules']
                self.length_lower_bound = arm['length_lower_bound']
                self.length_upper_bound = arm['length_upper_bound']
                self.parse_module_choices(arm)

            if 'mutation' in config:
                mutation = config['mutation']
                self.standard_deviation_length = mutation['standard_deviation_length']
                self.chance_module_drop = mutation['chance_module_drop']
                self.chance_module_add = mutation['chance_module_add']
                self.chance_type_mutation = mutation['chance_type_mutation']

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
                self.parse_workspace_parameters(morphevo)

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
                self.use_walls = rl['use_walls'] if 'use_walls' in rl else False

        def parse_workspace_parameters(self, config):
            if (    'workspace_type'        in config
                and 'workspace_cube_offset' in config
                and 'workspace_side_length' in config):

                self.workspace_parameters = WorkspaceParameters(config['workspace_type'],
                                                          tuple(config['workspace_cube_offset']),
                                                                config['workspace_side_length'])
            else:
                self.workspace_parameters = WorkspaceParameters()

        def parse_module_choices(self, config):
            self.module_choices = []

            if 'complex' in config['movements']:
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
