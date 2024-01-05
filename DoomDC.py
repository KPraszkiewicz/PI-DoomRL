import typing as t
import numpy as np
import vizdoom
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from stable_baselines3.common import vec_env
from vizdoom import Button
import itertools

import DoomEnv

class DoomDC(DoomEnv.DoomEnv):
    """Wrapper environment following OpenAI's gym interface for a VizDoom game instance."""

    def __init__(self,
                 game: vizdoom.DoomGame,
                 frame_processor: t.Callable,
                 frame_skip: int = 4,
                 combinated_buttons: bool = True):
        super().__init__(
            game,
            frame_processor,
            frame_skip,
            combinated_buttons
        )

    def step(self, action: int) -> t.Tuple[DoomEnv.Frame, int, bool, bool, t.Dict]:

        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        if(action == 4):
            reward += 0.1
        elif action == 5:
            reward += 0.1
        elif action == 3:
            reward += -0.1
        elif action == 2:
            reward += -0.1
        elif action == 0:
            reward += -0.1
        
        return self.state, reward, self.game.is_player_dead(), done, {}



