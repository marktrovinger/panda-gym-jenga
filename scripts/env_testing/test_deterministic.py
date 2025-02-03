import panda_gym
import gymnasium as gym
import panda_gym_jenga
import time


def main():
    # env = gym.make("JengaWall3Deterministic-v3", render_mode = "human", deterministic=True)
    env = gym.make(
        "JengaTower3Deterministic-v3", render_mode="human", deterministic=True
    )

    obs, done = env.reset()
    # action_space = env.action_space
    # env.action_space = Discrete(4)
    time = 0
    total_steps = 0
    # i = input("Press enter to end simulation.")
    for i in range(6):
        steps_taken = 0
        obs, reward, terminated, truncated, info = env.step(0)
        steps_taken = info["time_taken"]
        print(f"Moved to object in {steps_taken} steps.")
        obs, reward, terminated, truncated, info = env.step(2)
        steps_taken = reward * -1
        total_steps += steps_taken
        print(f"Grabbed object in {steps_taken} steps.")
        obs, reward, terminated, truncated, info = env.step(1)
        steps_taken = reward * -1
        total_steps += steps_taken
        print(f"Moved object to objective in {steps_taken} steps.")
        obs, reward, terminated, truncated, info = env.step(3)
        steps_taken = reward * -1
        total_steps += steps_taken
        print(f"Released object at objective in {steps_taken} steps.")
        total_steps += steps_taken

    print(f"Moved 4 objects in {total_steps} steps.")

    i = input("Press enter to end simulation.")
    env.close()


if __name__ == "__main__":
    main()
