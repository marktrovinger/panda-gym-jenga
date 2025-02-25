#from panda_gym_jenga.envs import Stack3
from stable_baselines3 import DQN, PPO
import gymnasium as gym
from rl_deterministic_actions.wrappers.deterministic_rl import DeterministicRLWrapper
from gymnasium.spaces.utils import flatten_space
import panda_gym_jenga
import wandb
import numpy as np
from wandb.integration.sb3 import WandbCallback
from rl_deterministic_actions.algorithms.q_learning import QAgent


def main():
    env = gym.make(
        "JengaTower3-v3", render_mode="human"
    )
    env = DeterministicRLWrapper(env)
    q = QAgent(2^env.observation_space.n, env.action_space.n)

    q.learn(env, 1000)


    #model_dqn = DQN(policy="MlpPolicy", env=env)
    #model_dqn.learn(100)

    #model = QAgent(state_space = env.observation_space, action_space=env.action_space)
    #model.train()

if __name__ == "__main__":
    main()
