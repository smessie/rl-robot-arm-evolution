from typing import Callable, Tuple

import numpy as np
import ray
from mlagents_envs.exception import UnityWorkerInUseException
from morphevo.genetic_encoding import Genome
from morphevo.workspace import Workspace

from environment.environment import SimEnv


@ray.remote(num_cpus=1)
class Evaluator:

    JOINT_ANGLE_STEP = 10

    def __init__(self, env_path: str, use_graphics: bool = True) -> None:
        self.env_path = env_path
        self.env = None
        self.joint_angles = None
        self.use_graphics = use_graphics

    def _initialize_environment(self, urdf: str, worker_id: int) -> np.ndarray:
        env_created = False
        while not env_created:
            try:
                env = SimEnv(self.env_path, urdf,
                             use_graphics=self.use_graphics,
                             worker_id=worker_id)
                env_created = True
            except UnityWorkerInUseException:
                worker_id = np.random.randint(low=1000, high=9000) + worker_id

        return env

    def _generate_joint_angles(self, angle_amount) -> np.ndarray:
        base_joint_angle_options = list(range(-180, 180, self.JOINT_ANGLE_STEP * 4))
        angle_options = list(range(0, 105, self.JOINT_ANGLE_STEP))

        joint_angles = []
        t_values = [1 for _ in range(angle_amount)]
        for base_joint_angle_option in base_joint_angle_options:
            self._generate_extra_angle(
                joint_angles, angle_options, [base_joint_angle_option], t_values, angle_amount - 1, 0
            )

        self.joint_angles = np.array(joint_angles)
        return self.joint_angles

    # pylint: disable=too-many-arguments
    def _generate_extra_angle(self, joint_angles, angle_options, joint_options_others,
                                    t_values,     angle_amount,  current_angle):
        for joint_option in angle_options[::t_values[current_angle]]:
            if current_angle == angle_amount - 1:
                joint_angles.append(joint_options_others[:] + [joint_option])
            else:
                self._generate_extra_angle(
                    joint_angles,
                    angle_options,
                    joint_options_others[:] + [joint_option],
                    t_values,
                    angle_amount,
                    current_angle + 1
                )
        t_values[current_angle] *= -1

    def _create_observation_parser(self, genome:Genome):

        def parse_observation(observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            joint_angles = observations[[i*4 for i in range(genome.amount_of_modules)]]
            ee_pos = observations[genome.amount_of_modules*4: genome.amount_of_modules*4 + 3]

            return joint_angles, ee_pos

        return parse_observation


    def _step_until_target_angles(self, target_angles: np.ndarray, workspace: Workspace,
                                  parse_observation: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> None:

        target_angles = ( target_angles // self.JOINT_ANGLE_STEP) * self.JOINT_ANGLE_STEP

        target_angles[0] = np.clip(target_angles[0], -180, 180)
        target_angles[1:] = np.clip(target_angles[1:], 0, 100)


        observations = self.env.get_current_state()
        prev_angles = np.ones(len(target_angles))
        actions = np.zeros(len(target_angles))

        done = False
        while not done:
            current_angles, ee_pos = parse_observation(observations)
            workspace.add_ee_position(ee_pos, current_angles)

            if abs(np.sum(current_angles - prev_angles)) < 0.01:
                break

            angle_diff = current_angles - target_angles
            actions[abs(angle_diff) < 5] = 0
            actions[angle_diff > 0] = -5
            actions[angle_diff < 0] = 5

            if np.count_nonzero(actions) == 0:
                done = True
            else:
                observations = self.env.step(actions)

            prev_angles = current_angles

    def eval_genome(self, genome: Genome) -> Genome:
        self.env = self._initialize_environment(genome.get_urdf(), genome.genome_id)
        self.env.reset()

        joint_angles = self._generate_joint_angles(genome.amount_of_modules)
        observation_parser = self._create_observation_parser(genome)

        for target_angles in joint_angles:
            self._step_until_target_angles(target_angles, genome.workspace, observation_parser)

        self.env.close()
        return genome
