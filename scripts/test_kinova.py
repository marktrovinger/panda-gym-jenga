from panda_gym_jenga.envs.robot.kinova import Kinova
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
import numpy as np

def main():
    sim = PyBullet(render_mode="human")
    robot = Kinova(sim, control_type="ee")

    for _ in range(50):
        robot.set_action(np.array([1.0, 1.0, 1.0, 1.0]))
        sim.step()

if __name__ == "__main__":
    main()