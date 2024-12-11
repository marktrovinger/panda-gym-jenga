import gymnasium as gym
import panda_gym_jenga
import matplotlib.pyplot as plt

env = gym.make("JengaPickAndPlace-v3", render_mode="rgb_array", renderer="OpenGL")

observation, info = env.reset()
image = env.render()
plt.plot(image)
plt.savefig("pickandplace.jpg")
env.close()

