from typing import Any, Dict, Tuple

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance
from panda_gym.envs.tasks.stack import Stack

class JengaSimplePickAndPlace(Task):
    """A simplified version of the pick and place task for Jenga blocks.
    """
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.1,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
        object_size="normal"
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = object_size
        if self.object_size == "large":
            self.extents = np.array([0.0381, 0.12065, 0.0254])
        else:
            pass
        #self.np_random = Task.
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.17, width=1.424, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="block1",
            half_extents=self.extents / 2,
            mass=2.0,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        self.sim.create_box(
            body_name="target1",
            half_extents=self.extents / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, self.extents[0] / 2, self.extents[2] / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )
    def get_obs(self) -> np.ndarray:
        # position, rotation of the block
        object1_position = np.array(self.sim.get_base_position("block1"))
        object1_rotation = np.array(self.sim.get_base_rotation("block1"))
        object1_velocity = np.array(self.sim.get_base_velocity("block1"))
        object1_angular_velocity = np.array(self.sim.get_base_angular_velocity("block1"))
        observation = np.concatenate(
            [
                object1_position,
                object1_rotation,
                object1_velocity,
                object1_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object1_position = self.sim.get_base_position("block1")
        #object3_position = self.sim.get_base_position("object3")
        return object1_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object1_position = self._sample_objects()
        self.sim.set_base_pose("target1", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("block1", object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        

    def _sample_goal(self) -> np.ndarray:
        goal1 = np.array([0.0, 0.0, self.extents[2] / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal1 += noise
        return goal1

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([0.0, 0.0, self.extents[2] / 2])
        noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object1_position += noise1
        # if distance(object1_position, object2_position) > 0.1:
        return object1_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        # must be vectorized !!
        d = distance(achieved_goal, desired_goal)
        return np.array((d < self.distance_threshold), dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array((d > self.distance_threshold), dtype=np.float32)
        else:
            return -d.astype(np.float32)
