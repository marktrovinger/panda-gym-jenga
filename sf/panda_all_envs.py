from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
        (
            "env",
            [
                "panda_reach",
                "panda_push",
                "panda_slide",
                "panda_pick_and_place",
                "panda_stack",
                "panda_flip",
            ],
        ),
    ]
)

_experiments = [
    Experiment(
        "panda_all_envs",
        "python train_panda.py --algo=APPO --with_wandb=True --wandb_tags panda",
        _params.generate_params(randomize=False),
    ),
]