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

class DoomHealthGathering(DoomEnv.DoomEnv):
    """Wrapper environment following OpenAI's gym interface for a VizDoom game instance."""

    def __init__(self,
                 game: vizdoom.DoomGame,
                 frame_processor: t.Callable,
                 frame_skip: int = 4,
                 combinated_buttons: bool = True,
                 rewards_extension: np.array = None):
        super().__init__(
            game,
            frame_processor,
            frame_skip,
            combinated_buttons,
            rewards_extension
        )
        self.last_health = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)

    def step(self, action: int) -> t.Tuple[DoomEnv.Frame, int, bool, bool, t.Dict]:

        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        health = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)
        reward = reward + max(health - self.last_health, 0)
        self.last_health = health

        return self.state, reward, self.game.is_player_dead(), done, {}

    def reset(self, seed=None, options=None) -> t.Tuple[DoomEnv.Frame, t.Dict]:

        self.game.new_episode()
        if seed:
            self.game.set_seed(seed)
        self.state = self._get_frame()
        self.last_health = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)

        return self.state, {}

