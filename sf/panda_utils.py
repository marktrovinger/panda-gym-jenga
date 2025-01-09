from typing import Optional

import gymnasium as gym

class PandaSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id

PANDA_ENVS = [
    PandaSpec("panda_reach", "PandaReach-v3"),
    PandaSpec("panda_push", "PandaPush-v3"),
    PandaSpec("panda_slide", "PandaSlide-v3"),
    PandaSpec("panda_pick_and_place", "PandaPickAndPlace-v3"),
    PandaSpec("panda_stack", "PandaStack-v3"),
    PandaSpec("panda_flip", "PandaFlip-v3"),
]

def panda_env_by_name(name):
    for cfg in PANDA_ENVS:
        if cfg.name == name:
            return cfg
        raise Exception("Unknown Panda env")
    
def make_panda_env(env_name, _cfg, _env_config, render_mode: Optional[str] = "rgb_array", **kwargs):
    panda_spec = panda_env_by_name(env_name)
    env = gym.make(panda_spec.env_id, render_mode="human")
    return env