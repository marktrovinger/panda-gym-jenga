from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
from panda_gym_jenga.envs.jenga_tasks import JengaStack3
import numpy as np
import matplotlib.pyplot as plt


class MyRobotTaskEnv(RobotTaskEnv):
    """My robot-task environment."""

    def __init__(self, render_mode):
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, base_position=np.array([-0.6, 0.0, 0.0]))
        task = JengaStack3(sim)
        super().__init__(robot, task)


def main():
    env = MyRobotTaskEnv(render_mode="rgb_array")

    observation, info = env.reset()
    image = env.render()
    plt.axis("off")
    plt.imshow(image)
    plt.show()

    

    for _ in range(1000):
        action = env.action_space.sample() # random action
        observation, reward, terminated, truncated, info = env.step(action)
        #env.render()

    if terminated or truncated:
        observation, info = env.reset()


if __name__ == "__main__":
    main()