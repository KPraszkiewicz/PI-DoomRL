#!/usr/bin/env python3

#####################################################################
# Example for running a vizdoom scenario as a gym env
#####################################################################

import gymnasium

from vizdoom import gymnasium_wrapper

from ViZDoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv  # noqa
# from ViZDoom.gym_wrapper.base_gym_env import VizdoomEnv  # noqa
from stable_baselines3 import PPO



if __name__ == "__main__":
    env = VizdoomEnv("scenarios/basic.cfg", render_mode="rgb_array")

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")

