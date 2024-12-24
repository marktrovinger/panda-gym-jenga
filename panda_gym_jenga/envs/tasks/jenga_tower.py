from typing import Any, Dict, Tuple

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance

class JengaTower(Task):
    """
    """
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.1,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
        object_size="large"
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = object_size
        if self.object_size == "large":
            # base layer points east/west, first layer points north/south
            self.extents_base = np.array([0.12065, 0.0381, 0.0254])
            self.extents_layer = np.array([0.0381, 0.12065, 0.0254])
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
            half_extents=self.extents_base / 2,
            mass=2.0,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        self.sim.create_box(
            body_name="target1",
            half_extents=self.extents_base / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, self.extents_base[2] / 2, self.extents_base[2] / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )
        self.sim.create_box(
            body_name="block2",
            half_extents=self.extents_base / 2,
            mass=2.0,
            position=np.array([1.0, 1.0, 1.0]),
            rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target2",
            half_extents=self.extents_base / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, -self.extents_base[1] / 2, self.extents_base[2] / 2]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="block3",
            half_extents=self.extents_layer / 2,
            mass=1.0,
            position=np.array([1.5, 1.5, 1.0]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target3",
            half_extents=self.extents_layer / 2,
            mass=0.0,
            ghost=True,
            position=np.array([self.extents_layer[1] / 2, 0.0, self.extents_layer[2] / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="block4",
            half_extents=self.extents_layer / 2,
            mass=2.0,
            position=np.array([1.0, 1.0, 0.0]),
            rgba_color=np.array([0.9, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target4",
            half_extents=self.extents_layer / 2,
            mass=0.0,
            ghost=True,
            position=np.array([-self.extents_layer[1] / 2, 0.0, self.extents_layer[2] / 2]),
            rgba_color=np.array([0.9, 0.9, 0.1, 0.3]),
        )
        

    def get_obs(self) -> np.ndarray:
        # position, rotation of the block
        object1_position = np.array(self.sim.get_base_position("block1"))
        object1_rotation = np.array(self.sim.get_base_rotation("block1"))
        object1_velocity = np.array(self.sim.get_base_velocity("block1"))
        object1_angular_velocity = np.array(self.sim.get_base_angular_velocity("block1"))
        object2_position = np.array(self.sim.get_base_position("block2"))
        object2_rotation = np.array(self.sim.get_base_rotation("block2"))
        object2_velocity = np.array(self.sim.get_base_velocity("block2"))
        object2_angular_velocity = np.array(self.sim.get_base_angular_velocity("block2"))
        object3_position = np.array(self.sim.get_base_position("block3"))
        object3_rotation = np.array(self.sim.get_base_rotation("block3"))
        object3_velocity = np.array(self.sim.get_base_velocity("block3"))
        object3_angular_velocity = np.array(self.sim.get_base_angular_velocity("block3"))
        object4_position = np.array(self.sim.get_base_position("block4"))
        object4_rotation = np.array(self.sim.get_base_rotation("block4"))
        object4_velocity = np.array(self.sim.get_base_velocity("block4"))
        object4_angular_velocity = np.array(self.sim.get_base_angular_velocity("block4"))
        

        observation = np.concatenate(
            [
                object1_position,
                object1_rotation,
                object1_velocity,
                object1_angular_velocity,
                object2_position,
                object2_rotation,
                object2_velocity,
                object2_angular_velocity,
                object3_position,
                object3_rotation,
                object3_velocity,
                object3_angular_velocity,
                object4_position,
                object4_rotation,
                object4_velocity,
                object4_angular_velocity,
            
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object1_position = self.sim.get_base_position("block1")
        object2_position = self.sim.get_base_position("block2")
        object3_position = self.sim.get_base_position("block3")
        object4_position = self.sim.get_base_position("block4")
        achieved_goal = np.concatenate((object1_position, object2_position, object3_position, object4_position))
        return achieved_goal

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object1_position, object2_position, object3_position, object4_position = self._sample_objects()
        self.sim.set_base_pose("target1", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.goal[3:6], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("block1", object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("block2", object2_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target3", self.goal[6:9], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target4", self.goal[9:], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("block3", object3_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("block4", object4_position, np.array([0.0, 0.0, 0.0, 1.0]))
        

    def _sample_goal(self) -> np.ndarray:
        goal1 = np.array([0.0, self.extents_base[1], self.extents_base[2] / 2])  # z offset for the cube center
        goal2 = np.array([0.0, -self.extents_base[1], self.extents_base[2] / 2])  # z offset for the cube center
        goal3 = np.array([self.extents_base[1], 0.0,  3 * self.extents_base[2] / 2])  # z offset for the cube center
        goal4 = np.array([-self.extents_base[1], 0.0, 3 * self.extents_base[2] / 2])  # z offset for the cube center
        return np.concatenate((goal1, goal2, goal3, goal4))

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([0.0, 0.0, self.extents_base[2] / 2])
        object2_position = np.array([0.0, 0.0, 3 * self.extents_base[2] / 2])
        object3_position = np.array([0.0, 0.0, 6 * self.extents_base[2] / 2])
        object4_position = np.array([0.0, 0.0, 9 * self.extents_base[2] / 2])
        noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object1_position += noise1
        object2_position += noise2
        noise3 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise4 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object3_position += noise3
        object4_position += noise4
        # if distance(object1_position, object2_position) > 0.1:
        return object1_position, object2_position, object3_position, object4_position

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
