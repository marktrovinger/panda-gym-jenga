import gymnasium as gym
import panda_gym
import panda_gym_jenga
import numpy as np



def main():
    env = gym.make("JengaPickAndPlace-v3", render_mode="human", robot="kinova")
    observation, info = env.reset()
    for _ in range(1000):
        current_position = observation["observation"][0:3]
        desired_position = observation["desired_goal"][0:3]
        action = 5.0 * (desired_position - current_position)
        action = np.append(action, 1.0)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()