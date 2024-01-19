import time
import cv2
from stable_baselines3 import DQN, PPO
import numpy as np
from DoomEnv import DoomEnv
from DoomWithBots import DoomWithBots
from env_utils import *
from utils import load_config

log_dir = "log_dc_mod1_PPO_test_1"
config = load_config(log_dir + "/config.ini")
record = True

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
    env = create_env(**env_args)

model = PPO.load(log_dir + "/best_model", env=env)

if record:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (320,  240))

en = model.get_env()
obs = en.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = en.step(action)

    frame = env.game.get_state().screen_buffer
    frame = np.transpose( frame, [1, 2, 0])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if record:
        out.write(frame)

    if reward[0] != 0.:
        print(reward)
    if done:
        break
        obs = en.reset()
    # time.sleep(0.1)
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
if record:
    out.release()