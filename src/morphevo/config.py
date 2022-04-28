from ray._private.ray_logging import configure_log_file
import yaml


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

    instance = None
    def __new__(cls, *args, **kwargs):
        if not Config.instance:
            Config.instance = Config.__Config(*args, **kwargs)
        return Config.instance

def set_config(config_file_name: str):
    return Config(config_file_name)

def get_config():
    return Config()

