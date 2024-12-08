from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
from stack3 import Stack3


class MyRobotTaskEnv(RobotTaskEnv):
    """My robot-task environment."""

    def __init__(self, render_mode):
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim)
        task = Stack3(sim)
        super().__init__(robot, task)


def main():
    env = MyRobotTaskEnv(render_mode="human")

    observation, info = env.reset()
    

    for _ in range(1000):
        action = env.action_space.sample() # random action
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

    if terminated or truncated:
        observation, info = env.reset()


if __name__ == "__main__":
    main()