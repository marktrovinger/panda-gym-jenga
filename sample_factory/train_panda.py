from typing import Optional
import argparse
import sys
import gymnasium as gym
import panda_gym

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from panda_utils import PANDA_ENVS, make_panda_env
from panda_params import panda_override_defaults

def register_panda_components():
    for env in PANDA_ENVS:
        register_env(env.name, make_panda_env)

# def add_custom_env_args(_env, p: argparse.ArgumentParser, evaluation=False):
#     # You can extend the command line arguments here
#     p.add_argument("--custom_argument", default="value", type=str, help="")


def parse_panda_args(argv=None, evaluation=False):
    # parse the command line arguments to build
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    #add_custom_env_args(partial_cfg.env, parser, evaluation=evaluation)
    panda_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

def main():
    """Script entry point."""
    register_panda_components()
    cfg = parse_panda_args()

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())