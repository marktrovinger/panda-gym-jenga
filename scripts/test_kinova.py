from panda_gym_jenga.envs.robot.kinova import Kinova
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
import numpy as np

import matplotlib.pyplot as plt

def main():
    sim = PyBullet(render_mode="human")
    robot = Kinova(sim, control_type="ee")
    i = input("Press enter to end simulation.")

    for _ in range(50):
        robot.set_action(np.array([1.0, 1.0, 1.0, 1.0]))
        #image = sim.render()
        sim.step()
        i = input("Press enter to end simulation.")
        #image = sim.render()
        #plt.imshow(image)

    

if __name__ == "__main__":
    main()