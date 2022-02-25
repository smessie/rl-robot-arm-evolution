from typing import Tuple
from environment.environment import SimEnv
from genetic_encoding import Genome
import numpy as np
from workspace import Workspace
from mlagents_envs.exception import UnityWorkerInUseException
import ray

@ray.remote(num_cpus=1)
class Evaluator:
    def __init__(self, env_path: str, use_graphics: bool = True) -> None:
        self.env_path = env_path
        self.env = None
        self.joint_angles = None
        self.use_graphics = use_graphics

    def _initialize_environment(self, urdf: str, worker_id: int) -> np.ndarray:
        env_created = False
        while not env_created:
            try:
                env = SimEnv(self.env_path, urdf, use_graphics=self.use_graphics, worker_id=worker_id)
                env_created = True
            except UnityWorkerInUseException:
                worker_id = np.random.randint(low=1000, high=9000) + worker_id

        return env

    def _parse_observation(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # [j0angel, j0x, j0y, j0z, ..., eex, eey, eez]
        joint_angles = observations[[0, 4, 8, 12]]
        ee_pos = observations[16:19]

        return joint_angles, ee_pos

    def _generate_joint_angle(self) -> np.ndarray:
        if self.joint_angles is not None:
            return self.joint_angles

        angle_step = self.env.JOINT_ANGLE_STEP
        joint0_angle_options = list(range(-180, 180, angle_step * 2))
        joint_angle_options = list(range(0, 105, angle_step))

        t1 = 1
        t2 = 1
        t3 = 1
        self.joint_angles = []
        for j0 in joint0_angle_options:
            for j1 in joint_angle_options[::t1]:
                for j2 in joint_angle_options[::t2]:
                    for j3 in joint_angle_options[::t3]:
                        self.joint_angles.append([j0, j1, j2, j3])
                    t3 *= -1
                t2 *= -1
            t1 *= -1
        
        self.joint_angles = np.array(self.joint_angles)
        return self.joint_angles

        

    def _step_until_target_angles(self, observations: np.ndarray, target_angles: np.ndarray,
                                  workspace: Workspace) -> np.ndarray:

        target_angles = (target_angles // self.env.JOINT_ANGLE_STEP) * self.env.JOINT_ANGLE_STEP

        target_angles[0] = np.clip(target_angles[0], -180, 180)
        target_angles[1:] = np.clip(target_angles[1:], 0, 100)

        done = False

        prev_angles = np.ones(4)
        actions = np.zeros(4)
        while not done:
            current_angles, ee_pos = self._parse_observation(observations)
            workspace.add_ee_position(ee_pos, current_angles)

            if abs(np.sum(current_angles - prev_angles)) < 0.01:
                break

            angle_diff = current_angles - target_angles
            actions[abs(angle_diff) < 5] = 0
            actions[angle_diff > 0] = -1
            actions[angle_diff < 0] = 1


            if np.count_nonzero(actions) == 0:
                done = True
            else:
                observations = self.env.step(actions)

            prev_angles = current_angles

        return observations


    def eval_genome(self, genome: Genome) -> Genome:
        self.env = self._initialize_environment(genome.get_urdf(), genome.genome_id)

        observations = self.env.reset()

        joint_angles = self._generate_joint_angle()

        for target_angles in joint_angles:
            observations = self._step_until_target_angles(observations, target_angles, genome.workspace)

        self.env.close()


        return genome



