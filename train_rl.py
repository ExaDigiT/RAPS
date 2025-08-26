import gym
from stable_baselines3 import PPO
from raps.envs.raps_env import RAPSEnv
from raps.system_config import get_system_config
from raps.sim_config import args, args_dict

config = get_system_config(args.system).get_legacy()
args_dict['config'] = config
args_dict['args'] = args

env = RAPSEnv(**args_dict)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save trained model
model.save("ppo_raps")
