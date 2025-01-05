import panda_gym
import gymnasium as gym
import panda_gym_jenga



def main():
    #env = gym.make("JengaSimplePickAndPlaceDeterministicEnv-v3", render_mode = "human")
    env = gym.make("JengaTowerDeterministic-v3", render_mode = "human", deterministic=True)
    #env = RecordVideo(env, video_folder="deterministic_testing", name_prefix="testing", episode_trigger=lambda x: True)
    
    obs, done = env.reset()
    #action_space = env.action_space
    #env.action_space = Discrete(4)
    time = 0
    steps_taken = 0
    obs, reward, terminated, truncated, info = env.step(0)
    steps_taken = reward * -1
    print(f"Moved to object in {steps_taken} steps.")
    obs, reward, terminated, truncated, info = env.step(2)
    steps_taken = reward * -1
    print(f"Grabbed object in {steps_taken} steps.")
    obs, reward, terminated, truncated, info = env.step(1)
    steps_taken = reward * -1
    print(f"Moved object to objective in {steps_taken} steps.")
    obs, reward, terminated, truncated, info = env.step(3)
    steps_taken = reward * -1
    print(f"Released object at objective in {steps_taken} steps.")
    
    for i in range(4):
        obs, reward, terminated, truncated, info = env.step(i)
        steps_taken += reward * -1
    env.close()

if __name__ == "__main__":
    main()
