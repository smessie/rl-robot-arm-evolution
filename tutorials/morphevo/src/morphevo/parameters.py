import yaml


class Parameters:

    def __init__(self, config_file_name: str) -> None:
        stream = open(config_file_name, 'r')
        evolution_config = yaml.load(stream, yaml.FullLoader)
        self.GENERATIONS = evolution_config['generations']
        self.MU = evolution_config['mu']
        self.LAMBDA = evolution_config['lambda']
