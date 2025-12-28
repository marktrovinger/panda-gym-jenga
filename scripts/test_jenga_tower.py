import gymnasium as gym
import panda_gym_jenga
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

def main():
    env = gym.make("JengaTower-v3", render_mode="rgb_array", renderer="OpenGL")
    env = RecordVideo(
        env,
        video_folder="tower-deterministic",
        name_prefix="pres",
        episode_trigger=lambda x: True
    )

    observation, info = env.reset()
    image = env.render()
    print(image.shape)
    plt.imshow(image)
    #plt.plot(image)
    #plt.savefig("JengaTower.jpg")
    plt.show()
    env.close()

if __name__ == "__main__":
    main()

