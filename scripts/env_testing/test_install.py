import panda_gym
import gymnasium as gym

from stable_baselines3 import DDPG

env = gym.make("PandaStack-v3")
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)
model.learn(30_000)