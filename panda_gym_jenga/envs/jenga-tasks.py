from typing import Optional

import numpy as np
from panda_gym.pybullet import PyBullet
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym_jenga.envs.tasks.jengapickandplace import JengaPickAndPlace
from panda_gym_jenga.envs.tasks.stack3 import Stack3



class JengaPickAndPlace(RobotTaskEnv):
    
    def __init__(
            self,
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_mode: str ="rgb_array",
            render_width = 720, 
            render_height = 480, 
            render_target_position = None, 
            render_distance = 1.4, 
            render_yaw = 45, 
            render_pitch = -30, 
            render_roll = 0,
            object_size: str = "large"
        ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([]), control_type=control_type)
        task = JengaPickAndPlace(sim, reward_type=reward_type, object_size=object_size)
        super().__init__(
            robot, 
            task, 
            render_width, 
            render_height, 
            render_target_position, 
            render_distance, 
            render_yaw, 
            render_pitch, 
            render_roll
        )