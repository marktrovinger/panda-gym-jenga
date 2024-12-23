import panda_gym
import gymnasium as gym
from gymnasium.spaces import Discrete
import panda_gym_jenga
from gymnasium.wrappers import RecordVideo



def main():
    env = gym.make("JengaPickAndPlaceDeterministic-v3", render_mode = "human")
    #env = RecordVideo(env, video_folder="deterministic_testing", name_prefix="testing", episode_trigger=lambda x: True)
    
    obs, done = env.reset()
    action_space = env.action_space
    env.action_space = Discrete(4)
    print(f"Action space: {env.action_space}")
    obs, reward, terminated, truncated, info = env.step(0)
    #print(obs)
    print(f"Reward = {reward}")
    #env.step(2)
    env.step(2)
    print(f"Reward = {reward}")
    env.step(1)
    print(f"Reward = {reward}")
    env.close()

if __name__ == "__main__":
    main()
