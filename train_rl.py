from stable_baselines3 import PPO
from raps.envs.raps_env import RAPSEnv
from raps.system_config import get_system_config
from raps.sim_config import args, args_dict

config = get_system_config(args.system).get_legacy()
args_dict['config'] = config
args_dict['args'] = args

env = RAPSEnv(**args_dict)

model = PPO(
    "MlpPolicy",
    env,
    n_steps=512,         # shorter rollouts (quicker feedback loop)
    batch_size=128,      # must divide n_steps evenly
    n_epochs=10,         # # of minibatch passes per update
    gamma=0.99,          # discount (keeps long-term credit)
    learning_rate=3e-4,  # default Adam lr, can try 1e-4 if unstable
    ent_coef=0.01,       # encourage exploration
    verbose=1,
)

model.learn(total_timesteps=10000)

# Save trained model
model.save("ppo_raps")
