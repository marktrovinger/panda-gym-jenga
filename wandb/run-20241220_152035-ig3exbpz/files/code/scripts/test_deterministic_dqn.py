#from panda_gym_jenga.envs import Stack3
from stable_baselines3 import DQN
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import ActionWrapper
import panda_gym_jenga
import wandb
from wandb.integration.sb3 import WandbCallback

class JengaActionWrapper(ActionWrapper):
    def __init__(self, env):
        self.action_space = gym.spaces.Discrete(4,)
        super().__init__(env)

    
config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 100,
        "env_name": "JengaSimplePickAndPlaceDeterministic-v3",
    }
def make_env():
    env = gym.make(config["env_name"])
    env = JengaActionWrapper(env)
    print(f"Action Space: `{env.action_space}")
    #env = Monitor(env)  # record stats such as returns
    return env

def main():
    goal_selection_strategy = "future"

    run = wandb.init(
        project="parameter_testing",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    env = make_env()
    model = DQN(
        config["policy_type"], 
        env=env,
        verbose=1, 
        tensorboard_log=f"runs/{run.id}"
    )
    
    model.learn(
        config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2
        ),
    )
    env.close()
    run.finish()


if __name__ == "__main__":
    main()
