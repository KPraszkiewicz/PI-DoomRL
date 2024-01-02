import cv2
import time
from stable_baselines3 import DQN, PPO
import numpy as np
import DoomHealthGathering
from utils import load_config
from env_utils import *
from DoomEnv import DoomEnv
from DoomWithBots import DoomWithBots

env_args = {
    'scenario': "health_gathering",
    'visible': True,
    'EnvClass': DoomEnv,
    'frame_skip': 1,
    'frame_processor': lambda frame: cv2.resize(
        frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA),
    'combinated_buttons': False,
    # 'n_bots': 8
}

env = create_env(**env_args )
# env = env_with_bots(**env_args )


obs = env.reset()
for i in range(10000):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(reward)
    # VecEnv resets automatically
    if done:
      obs = env.reset()
    time.sleep(100)
