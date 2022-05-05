from typing import Optional

from morphevo.genetic_encoding import Genome
from rl.dqn import DQN


class Arm:
    def __init__(self, parent_genome: Optional[Genome] = None) -> None:
        self.urdf_path: str = ""
        self.genome: Genome = parent_genome if parent_genome else Genome()
        self.rl_model: DQN = None
        self.success_rate: float = None

    @property
    def urdf(self):
        return self.genome.get_urdf()
