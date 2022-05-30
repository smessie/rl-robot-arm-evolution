from typing import Optional

from morphevo.genome import Genome
from rl.dqn import DQN


class Arm:
    """! Class that represents an arm.
    """
    def __init__(self, genome: Optional[Genome] = None) -> None:
        """! The Arm class initializer
        @param genome Representation of the arm as a phenotype
        """
        self.urdf_path: str = ""
        self.genome: Genome = Genome(genome) if genome else Genome()
        self.rl_model: DQN = None
        self.success_rate: float = None

    @property
    def urdf(self):
        return self.genome.get_urdf()
