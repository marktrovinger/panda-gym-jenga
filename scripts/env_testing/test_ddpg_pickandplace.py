#from panda_gym_jenga.envs import Stack3
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
import panda_gym_jenga
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 1e7,
        "env_name": "JengaPickAndPlace-v3",
    }
def make_env():
    env = make_vec_env(config["env_name"], n_envs=64)
    env = VecMonitor(env)  # record stats such as returns
    return env

def main():
    goal_selection_strategy = "future"

    run = wandb.init(
        project="ddpg_experiments",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    env = make_env()
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 100000 == 0,
        video_length=200,
        name_prefix="ddpg_pickandplace"
    )
    model = DDPG(
        config["policy_type"], 
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy,
        ),
        verbose=1, 
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs={"net_arch": [400,300]}
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
