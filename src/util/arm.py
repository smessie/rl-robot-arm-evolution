from typing import Optional

from morphevo.directed_graph_encoding import Genome
from rl.dqn import DQN


class Arm:
    def __init__(self, genome: Optional[Genome] = None) -> None:
        self.urdf_path: str = ""
        self.genome: Genome = genome if genome else Genome()
        self.rl_model: DQN = None
        self.success_rate: float = None

    @property
    def urdf(self):
        return self.genome.get_urdf()
