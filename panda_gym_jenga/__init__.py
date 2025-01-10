import os

from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

ENV_IDS = []

for task in ["Stack3", "PickAndPlace", "PickAndPlaceDeterministic" "SimplePickAndPlace", "SimplePickAndPlaceDeterministic", "Tower", "TowerDeterministic", "Tower3Deterministic", "Tower3", "Wall3"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            env_id = f"Jenga{task}{control_suffix}{reward_suffix}-v3"

            register(
                id=env_id,
                entry_point=f"panda_gym_jenga.envs:Jenga{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type},
                max_episode_steps=350 if task == "Stack3" or "Tower" in task or "Wall" in task else 50,
            )
            ENV_IDS.append(env_id)
