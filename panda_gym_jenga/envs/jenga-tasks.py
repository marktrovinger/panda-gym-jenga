from typing import Optional

import numpy as np
from panda_gym.pybullet import PyBullet
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym_jenga.envs.tasks.jengapickandplace import JengaPickAndPlace
from panda_gym_jenga.envs.tasks.stack3 import Stack3



class JengaPickAndPlace(RobotTaskEnv):
    """Pick and place task for Jenga blocks with the Panda robot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters
    """
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

class JengaStack3(RobotTaskEnv):
    """Stack task with 3 blocks for the Panda robot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """
    def __init__(
            self,
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_mode: str ="rgb_array",
            render_width = 720, 
            render_height = 480, 
            render_target_position: Optional[np.ndarray] = None, 
            render_distance = 1.4, 
            render_yaw = 45, 
            render_pitch = -30, 
            render_roll = 0,
        ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Stack3(sim, reward_type=reward_type)
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