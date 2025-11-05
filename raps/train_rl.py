from raps.sim_config import SingleSimConfig, SIM_SHORTCUTS
from raps.utils import SubParsers, pydantic_add_args, read_yaml_parsed


def train_rl_add_parser(subparsers: SubParsers):
    parser = subparsers.add_parser("train-rl", description="""
        Example usage:
            raps train-rl --system mit_supercloud/part-gpu -f /opt/data/mit_supercloud/202201
    """)
    parser.add_argument("config_file", nargs="?", default=None, help="""
        YAML sim config file, can be used to configure an experiment instead of using CLI
        flags. Pass "-" to read from stdin.
    """)
    model_validate = pydantic_add_args(parser, SingleSimConfig, model_config={
        "cli_shortcuts": SIM_SHORTCUTS,
    })

    def impl(args):
        model = model_validate(args, read_yaml_parsed(SingleSimConfig, args.config_file))
        model.scheduler = "rl"
        train_rl(model)
    parser.set_defaults(impl=impl)


def train_rl(rl_config: SingleSimConfig):
    from stable_baselines3 import PPO
    from raps.envs.raps_env import RAPSEnv

    args_dict = rl_config.get_legacy_args_dict()
    config = rl_config.system_configs[0].get_legacy()
    args_dict['config'] = config
    args_dict['args'] = rl_config.get_legacy_args()

    env = RAPSEnv(rl_config)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,         # shorter rollouts (quicker feedback loop)
        batch_size=128,      # must divide n_steps evenly
        n_epochs=10,         # of minibatch passes per update
        gamma=0.99,          # discount (keeps long-term credit)
        learning_rate=3e-4,  # default Adam lr, can try 1e-4 if unstable
        ent_coef=0.01,       # encourage exploration
        verbose=1,
        tensorboard_log="./ppo_raps_logs/"
    )

    model.learn(total_timesteps=10000, tb_log_name="ppo_raps")

    # Output stats
    stats = env.get_stats()
    print(stats)

    # Save trained model
    model.save("ppo_raps")
