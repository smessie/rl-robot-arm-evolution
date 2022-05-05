from morphevo.genetic_encoding import Genome
from rl.dqn import DQN


class Arm:
    urdf_path: str = ""
    genome: Genome = None
    rl_model: DQN = None
    success_rate: float = None
