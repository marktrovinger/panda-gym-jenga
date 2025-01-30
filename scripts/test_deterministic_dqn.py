#from panda_gym_jenga.envs import Stack3
from stable_baselines3 import DQN
import gymnasium as gym
from rl_deterministic_actions.wrappers.deterministic_rl import DeterministicRLWrapper
from gymnasium.spaces import MultiDiscrete
import panda_gym_jenga
import wandb
import numpy as np
from wandb.integration.sb3 import WandbCallback

def main():
    env = gym.make(
        "JengaTower3Deterministic-v3", render_mode="human", deterministic=True
    )
    env = DeterministicRLWrapper(env)

    obs, _ = env.reset()
    action = np.array([0, 0], dtype=np.int32)
    action[0] = 0
    action[1] = 4
    obs, reward, terminated, truncated, info = env.step(action)
    action[0] = 1
    obs, reward, terminated, truncated, info = env.step(action)

if __name__ == "__main__":
    main()
