import panda_gym
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv

from wandb.integration.sb3 import WandbCallback
import wandb
config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 1e5,
        "env_name": "PandaStack-v3",
    }
def make_env():
    env = make_vec_env(config["env_name"], n_envs=4, vec_env_cls=SubprocVecEnv)
    env = Monitor(env)  # record stats such as returns
    return env

def main():

    run = wandb.init(
        project="parameter_testing",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    env = make_env()
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=200,
    )
    model = PPO(
        config["policy_type"], 
        env=env,
        verbose=1, 
        tensorboard_log=f"runs/{run.id}",
        device="cpu"
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

