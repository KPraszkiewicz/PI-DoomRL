import cv2
import gymnasium as gym
import DoomEnv
import shutil
import os
import sys
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from DoomHealthGathering import DoomHealthGathering

from utils import load_config

config = load_config("config.ini")

name = config.get('general', 'name')
version = config.get('general', 'version')
# parameters
algorithm = config.get('learning', 'algorithm')
frame_skip = config.getint('learning', 'frame_skip')
rewards_extension = config.getboolean('learning', 'rewards_extension')
scenario = config.get('game', 'scenario')
combinated_buttons = config.getboolean('game', 'combinated_buttons')
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

DQN_parameters = {
    'learning_rate': config.getfloat('DQN', 'learning_rate'),
    'policy': config.get('DQN', 'policy'),
    'buffer_size': config.getint('DQN', 'buffer_size'),
    'learning_starts': config.getint('DQN', 'learning_starts'),
    'batch_size': config.getint('DQN', 'batch_size'),
    'tau': config.getfloat('DQN', 'tau'),
    'gamma': config.getfloat('DQN', 'gamma'),
    'train_freq': config.getint('DQN', 'train_freq'),
    'gradient_steps': config.getint('DQN', 'gradient_steps'),
    'target_update_interval': config.getint('DQN', 'target_update_interval'),
    'exploration_fraction': config.getfloat('DQN', 'exploration_fraction'),
    'exploration_initial_eps': config.getfloat('DQN', 'exploration_initial_eps'),
    'exploration_final_eps': config.getfloat('DQN', 'exploration_final_eps'),
    'max_grad_norm': config.getfloat('DQN', 'max_grad_norm'),
    'stats_window_size': config.getint('DQN', 'stats_window_size'),
    #'tensorboard_log': config.get('DQN', 'tensorboard_log'), #TODO: Optional[str]
}
log_dir = f"log_{scenario}_{algorithm}_{name}_{version}"
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

rewards = None
if rewards_extension:
    rewards = np.array([])

env_args = {
    'scenario': scenario,
    'visible': False,
    'EnvClass': DoomHealthGathering,
    'frame_skip': frame_skip,
    'frame_processor': lambda frame: cv2.resize(
        frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA),
    'combinated_buttons': combinated_buttons,
    'rewards_extension': None
}

vec_env = DoomEnv.create_vec_env(n_envs, **env_args)
eval_vec_env = DoomEnv.create_vec_env(1, **env_args)

if algorithm == 'PPO':
    model = PPO(env=vec_env, verbose=1,
                **PPO_parameters
                )
elif algorithm == 'DQN':
    model = DQN(env=vec_env, verbose=1,
                **DQN_parameters
                )
else:
    exit()

eval_callback = EvalCallback(eval_vec_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=200,
                             deterministic=True, render=False, n_eval_episodes=10)

model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
