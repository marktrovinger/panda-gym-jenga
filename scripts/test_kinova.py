from panda_gym_jenga.envs.robot.kinova import Kinova
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
import numpy as np

import matplotlib.pyplot as plt

def main():
    sim = PyBullet(render_mode="human")
    robot = Kinova(sim, control_type="ee")

    for _ in range(50):
        robot.set_action(np.array([1.0, 1.0, 1.0, 1.0]))
        #image = sim.render()
        sim.step()
        #image = sim.render()
        #plt.imshow(image)

    i = input("Press enter to end simulation.")

if __name__ == "__main__":
    main()