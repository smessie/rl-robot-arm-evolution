from typing import Callable, Tuple

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

    def __init__(self, env_path: str, use_graphics: bool = True, sample_size: int = 100) -> None:
        self.env_path = env_path
        self.env = None
        self.joint_angles = None
        self.use_graphics = use_graphics
        self.sample_size = sample_size

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

    def _generate_joint_angles_samples(self, angle_amount, samples_amount) -> np.ndarray:
        base_joint_angle_options = list(range(0, 360, self.JOINT_ANGLE_STEP * 4))
        angle_options = list(range(0, 105, self.JOINT_ANGLE_STEP))

        samples = []
        for _ in range(samples_amount):
            anchor_angle = [np.random.choice(base_joint_angle_options)]
            other_angles = [np.random.choice(angle_options) for _ in range(angle_amount - 1)]
            samples.append(np.array(anchor_angle + other_angles))

        return np.array(samples)

    def _create_observation_parser(self):

        def parse_observation(observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # [j0angel, j0x, j0y, j0z, ..., eex, eey, eez]
            last_joint_index = self.env.joint_amount * 4
            joint_angles = observations[[i * 4 for i in range(self.env.joint_amount)]]
            ee_pos = observations[last_joint_index:last_joint_index + 3]

            return joint_angles, ee_pos

        return parse_observation

    def _step_until_target_angles(self, target_angles: np.ndarray, workspace: Workspace,
                                  parse_observation: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]) \
            -> None:

        target_angles = (target_angles // self.JOINT_ANGLE_STEP) * self.JOINT_ANGLE_STEP

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
            actions[angle_diff > 0] = -1
            actions[angle_diff < 0] = 1
            actions[abs(angle_diff) <= 5] = 0

            if np.count_nonzero(actions) == 0:
                done = True
            else:
                observations = self.env.step(actions)

            prev_angles = current_angles

    def _step_random_directions(self, joint_amount: int, workspace: Workspace,
                                parse_observation: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> None:
        observations = self.env.get_current_state()

        for _ in range(self.EVALUATIONS_AMOUNT):
            self.env.reset()

            for i in range(self.STEPS_PER_EVALUATION):
                actions = np.random.choice([0, -5, 5], joint_amount)
                observations = self.env.step(actions, return_observations=i + 1 == self.STEPS_PER_EVALUATION)

            current_angles, ee_pos = parse_observation(observations)
            workspace.add_ee_position(ee_pos, current_angles)

    def eval_arm(self, arm: Arm, config) -> Arm:
        self.env = self._initialize_environment(arm.genome.get_urdf(), arm.genome.genome_id)
        self.env.reset()

        joint_angles = self._generate_joint_angles_samples(self.env.joint_amount, config.sample_size)
        observation_parser = self._create_observation_parser()

        selected_joint_angles_indices = np.random.choice(joint_angles.shape[0], self.sample_size)
        selected_joint_angles = joint_angles[selected_joint_angles_indices, :]

        for target_angles in selected_joint_angles:
            self._step_until_target_angles(target_angles, arm.genome.workspace, observation_parser)

        self.env.close()
        return arm
