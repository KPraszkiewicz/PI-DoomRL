import cv2
from stable_baselines3 import DQN, PPO
import numpy as np
import DoomEnv
from utils import load_config


env_args = {
    'scenario': "take_cover",
    'visible': True,
    'frame_skip': 1,
    'frame_processor': lambda frame: cv2.resize(
        frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
}

env = DoomEnv.create_env(**env_args )


obs = env.reset()
for i in range(10000):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
