from typing import Optional

import numpy as np


from panda_gym.envs.core import RobotTaskEnv
from task.stack3 import Stack3
from panda_gym.pybullet import PyBullet


class JengaStack3(RobotTaskEnv):
    def __init__(
            self,
            render_mode: str = "rgb_array",
            reward_type: str = "sparse",
    ):
        pass