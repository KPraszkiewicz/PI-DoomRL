import cv2
from stable_baselines3 import PPO

import DoomEnv

log_dir = "log_basic_PPO_1/"

env_args = {
    'scenario': "basic",
    'frame_skip': 1,
    'frame_processor': lambda frame: cv2.resize(
        frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
}

env = DoomEnv.create_env(**env_args )
model = PPO.load(log_dir + "best_model.zip", env=env)


env = model.get_env()
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()