#from panda_gym_jenga.envs import Stack3
from stable_baselines3 import DDPG
import gymnasium as gym
import panda_gym_jenga


def main():
    env = gym.make("JengaStack3-v3")
    model = DDPG("MultiInputPolicy", env=env, verbose=1)
    model.learn(20_000)



if __name__ == "__main__":
    main()
