from morphevo.genetic_encoding import Genome
from rl.dqn import DQN


class Arm:
    def __init__(self):
        self.urdf_path: str = ""
        self.genome: Genome = None
        self.rl_model: DQN
        self.success_rate: float = None
