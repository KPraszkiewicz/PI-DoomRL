import cv2
import gymnasium as gym
import DoomEnv
import shutil
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from utils import load_config

config = load_config("config.ini")

# parameters
algorithm = config.get('learning', 'algorithm')
frame_skip = config.getint('learning', 'frame_skip')
scenario = config.get('game', 'scenario')
total_timesteps = config.getint('learning', 'total_timesteps')
n_envs = config.getint('learning', 'n_envs')
continue_learning = config.getboolean('learning', 'continue')

PPO_parameters = {
    'learning_rate': config.getfloat('PPO', 'learning_rate'),
    'policy': config.get('PPO', 'policy'),
    'batch_size': config.getint('PPO', 'batch_size'),
    'gamma': config.getfloat('PPO', 'gamma'),
    'n_steps': config.getint('PPO', 'n_steps'),
    'n_epochs': config.getint('PPO', 'n_epochs'),
    'gae_lambda': config.getfloat('PPO', 'gae_lambda'),
    'clip_range': config.getfloat('PPO', 'clip_range'),
    'clip_range_vf': None
}

log_dir = f"log_{scenario}_{algorithm}_test"
print(log_dir)

if not os.path.exists(log_dir):
    try:
        os.mkdir(log_dir)
    except OSError as error:  # TODO: lepsza obsługa
        print(error)
elif continue_learning:
    pass
else:
    sys.exit("Plik istnieje i opcja 'continue' jest wyłączona")



shutil.copy("config.ini", log_dir)

env_args = {
    'scenario': scenario,
    'visible': False,
    'frame_skip': frame_skip,
    'frame_processor': lambda frame: cv2.resize(
        frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
}

vec_env = DoomEnv.create_vec_env(n_envs, **env_args)
eval_vec_env = DoomEnv.create_vec_env(1, **env_args)

model = PPO(env=vec_env, verbose=1,
            **PPO_parameters
            )

eval_callback = EvalCallback(eval_vec_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=200,
                             deterministic=True, render=False, n_eval_episodes=10)

model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
