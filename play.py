import cv2
from stable_baselines3 import DQN, PPO
import numpy as np
import DoomEnv
from utils import load_config

log_dir = "log_dtc_6btn_PPO_cnn_test_less_actions"
config = load_config(log_dir + "/config.ini")

# parameters
scenario = config.get('game', 'scenario')
compress_buttons = config.getboolean('game', 'compress_buttons')

env_args = {
    'scenario': scenario,
    'visible': True,
    'frame_skip': 1,
    'frame_processor': lambda frame: cv2.resize(
        frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA),
    'compress_buttons': compress_buttons
}

env = DoomEnv.create_env(**env_args )
model = PPO.load(log_dir + "/best_model", env=env)


env = model.get_env()
obs = env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
