import cv2
from stable_baselines3 import DQN, PPO
import numpy as np
from DoomEnv import DoomEnv
from DoomWithBots import DoomWithBots
from env_utils import *
from utils import load_config

log_dir = "log_deathmatch_simple_PPO_test_1"
config = load_config(log_dir + "/config.ini")

# parameters
scenario = config.get('game', 'scenario')
combinated_buttons = config.getboolean('game', 'combinated_buttons')
n_bots = config.getint('game', 'n_bots')

env_args = {
    'scenario': scenario,
    'visible': True,
    'frame_skip': 1,
    'frame_processor': lambda frame: cv2.resize(
        frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA),
    'combinated_buttons': combinated_buttons
}

if n_bots > 0:
    env_args['EnvClass'] = DoomWithBots
    env_args['n_bots'] = n_bots
    env = env_with_bots(**env_args)
else:
    env_args['EnvClass'] = DoomEnv
    env = env_with_bots(**env_args)

model = PPO.load(log_dir + "/best_model", env=env)


env = model.get_env()
obs = env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
