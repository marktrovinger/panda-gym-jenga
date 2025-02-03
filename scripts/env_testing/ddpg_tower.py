from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
#from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import panda_gym_jenga
import wandb
from wandb.integration.sb3 import WandbCallback
config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 5e7,
        "env_name": "JengaTower-v3",
    }
def make_env():
    env = make_vec_env(config["env_name"], n_envs=80, vec_env_cls=SubprocVecEnv)
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
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 200000 == 0,
        video_length=200,
    )
    model = DDPG(
        config["policy_type"], 
        env=env,
        learning_starts=1000,
        tau=0.95,
        train_freq=1000,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy,
        ),
        verbose=1, 
        policy_kwargs= {
            "net_arch":[256, 256, 256],
        },
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
