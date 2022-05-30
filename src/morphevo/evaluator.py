from typing import Tuple

import numpy as np
import ray
from mlagents_envs.exception import UnityWorkerInUseException

from environment.environment import SimEnv
from morphevo.workspace import Workspace
from util.arm import Arm


@ray.remote(num_cpus=1)
class Evaluator:
    """! Class that is used to evaluate an arm.
    """
    def __init__(self, env_path: str, use_graphics: bool = True, sample_size: int = 100) -> None:
        """!
        @param env_path Path to the unity executable.
        @param use_graphics To show the environment or not.
        @param sample_size Amount of samples that will be used to evaluate arm.
        """
        self.env_path = env_path
        self.env = None
        self.use_graphics = use_graphics
        self.sample_size = sample_size

    def _initialize_environment(self, urdf: str, worker_id: int) -> SimEnv:
        """! Initialize the unity environment.
        @param urdf The urdf of the arm that will be used in the environment.
        @param worker_id The id for the environment worker.
        @return The newly created environment.
        """
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
        """! Generate random angle configurations.
        @param angle_amount The amount of joints an arm has.
        @param samples_amount The amount of sample configurations you need.
        @return Random angleconfiguration samples.
        """
        base_joint_angle_options = list(range(0, 360, self.env.JOINT_ANGLE_STEP * 4))
        angle_options = list(range(0, 105, self.env.JOINT_ANGLE_STEP))

        samples = []
        for _ in range(samples_amount):
            anchor_angle = [np.random.choice(base_joint_angle_options)]
            other_angles = [np.random.choice(angle_options) for _ in range(angle_amount - 1)]
            samples.append(np.array(anchor_angle + other_angles))

        return np.array(samples)

    def parse_observation(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """! Function to parse an observation from unity
        @param observations The observation form unity
        @return The joint angles configuration and the endeffector position
        """
        # [j0angel, j0x, j0y, j0z, ..., eex, eey, eez]
        last_joint_index = self.env.joint_amount * 4
        joint_angles = observations[[i * 4 for i in range(self.env.joint_amount)]]
        ee_pos = observations[last_joint_index:last_joint_index + 3]

        return joint_angles, ee_pos

    def _step_until_target_angles(self, target_angles: np.ndarray, workspace: Workspace) -> None:
        """! Let the arm step to given angleconfiguration. This will be done in the unity environment. This
        will stop when target reached or when no progress is made.
        @param target_angles The target angleconfiguration.
        @param workspace The workspace of the arm, used to register the steps.
        """

        target_angles = (target_angles // self.env.JOINT_ANGLE_STEP) * self.env.JOINT_ANGLE_STEP

        target_angles[0] = np.clip(target_angles[0], -180, 180)
        target_angles[1:] = np.clip(target_angles[1:], 0, 100)

        observations = self.env.get_current_state()
        prev_angles = np.ones(len(target_angles))
        actions = np.zeros(len(target_angles))

        done = False
        while not done:
            current_angles, ee_pos = self.parse_observation(observations)
            workspace.add_end_effector_position(ee_pos, current_angles)

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

    def eval_arm(self, arm: Arm) -> Arm:
        """! Evaluate an arm by sampling an amount of angleconfigurations which will be tried to
        reach.
        @param arm The arm that will be evaluated.
        @return The evaluated arm.
        """
        self.env = self._initialize_environment(arm.genome.get_urdf(), arm.genome.genome_id)
        self.env.reset()

        target_angles = self._generate_joint_angles_samples(self.env.joint_amount, self.sample_size)

        for target_angle in target_angles:
            self._step_until_target_angles(target_angle, arm.genome.workspace)

        self.env.close()
        return arm
