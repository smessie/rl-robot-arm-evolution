from typing import Callable, List, Tuple

import numpy as np
import ray
from mlagents_envs.exception import UnityWorkerInUseException

from environment.environment import SimEnv
from morphevo.workspace import Workspace
from util.arm import Arm


@ray.remote(num_cpus=1)
class Evaluator:
    JOINT_ANGLE_STEP = 10
    EVALUATIONS_AMOUNT = 1000
    STEPS_PER_EVALUATION = 20

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
            except (UnityWorkerInUseException, OverflowError) as _:
                worker_id = (np.random.randint(low=1000, high=9000) + worker_id) % 65535

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
                              t_values, angle_amount, current_angle):
        for joint_option in angle_options[::t_values[current_angle]]:
            if current_angle >= angle_amount - 1:
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

    def _create_observation_parser(self):

        def parse_observation(observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
            # [j0angel, j0x, j0y, j0z, ..., eex, eey, eez]
            last_joint_index = self.env.joint_amount * 4
            joint_angles = observations[[i * 4 for i in range(self.env.joint_amount)]]
            ee_pos = observations[last_joint_index:last_joint_index + 3]
            joint_positions = []
            i = 1
            while i < last_joint_index:
                joint_positions.append(tuple(observations[i:i + 3]))
                i += 3

            return joint_angles, ee_pos, joint_positions

        return parse_observation

    def _step_random_directions(self, joint_amount: int, workspace: Workspace,
                                parse_observation: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, List]]) -> None:
        observations = self.env.get_current_state()

        for _ in range(self.EVALUATIONS_AMOUNT):
            self.env.reset()

            for i in range(self.STEPS_PER_EVALUATION):
                actions = np.random.choice([0, -5, 5], joint_amount)
                observations = self.env.step(actions, return_observations=i + 1 == self.STEPS_PER_EVALUATION)

            current_angles, ee_pos, joint_positions = parse_observation(observations)
            workspace.add_ee_position(ee_pos, current_angles, joint_positions)

    def eval_arm(self, arm: Arm) -> Arm:
        self.env = self._initialize_environment(arm.genome.get_urdf(), arm.genome.genome_id)
        self.env.reset()

        observation_parser = self._create_observation_parser()

        self._step_random_directions(self.env.joint_amount, arm.genome.workspace, observation_parser)

        self.env.close()
        return arm
