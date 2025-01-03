#from panda_gym_jenga.envs import Stack3
from stable_baselines3 import DQN
import gymnasium as gym
from gymnasium import ActionWrapper
import panda_gym_jenga
import wandb
from wandb.integration.sb3 import WandbCallback

class JengaActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3,)

    def action(self, action):
        return action

config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 100,
        "env_name": "JengaSimplePickAndPlaceDeterministic-v3",
    }
def make_env():
    env = gym.make(config["env_name"])
    env = JengaActionWrapper(env)
    return env

def main():

    env = make_env()
    print(f"Action Space:{env.action_space}")
    model = DQN(
        config["policy_type"], 
        env=env,
        verbose=1,
    )
    
    model.learn(
        config["total_timesteps"],
    )
    env.close()


if __name__ == "__main__":
    main()
