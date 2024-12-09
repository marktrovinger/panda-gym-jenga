#from panda_gym_jenga.envs import Stack3
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import gymnasium as gym
import panda_gym_jenga
import wandb
from wandb.integration.sb3 import WandbCallback
config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 30000,
        "env_name": "JengaStack3-v3",
    }
def make_env(config):
    env = gym.make(config["env_name"])
    env = Monitor(env)  # record stats such as returns
    return env

def main():

    run = wandb.init(
        project="wandb_testing",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=200,
    )
    model = DDPG(config["policy_type"], env=env, verbose=1, tensorboard_log=f"runs/{run.id}")
    
    model.learn(
        config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2
        ),
    )
    run.finish()


if __name__ == "__main__":
    main()
