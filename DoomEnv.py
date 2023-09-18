import typing as t

import numpy as np
import vizdoom
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from stable_baselines3.common import vec_env

Frame = np.ndarray


class DoomEnv(Env):
    """Wrapper environment following OpenAI's gym interface for a VizDoom game instance."""

    def __init__(self,
                 game: vizdoom.DoomGame,
                 frame_processor: t.Callable,
                 frame_skip: int = 4):
        super().__init__()

        # Determine action space
        self.action_space = spaces.Discrete(game.get_available_buttons_size())

        # Determine observation space
        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        new_h, new_w, new_c = frame_processor(np.zeros((h, w, c))).shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8)

        # Assign other variables
        self.game = game
        self.possible_actions = np.eye(self.action_space.n).tolist()  # VizDoom needs a list of buttons states.
        self.frame_skip = frame_skip
        self.frame_processor = frame_processor

        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.state = self.empty_frame
        self.render_mode = "rgb_array"

    def step(self, action: int) -> t.Tuple[Frame, int, bool, bool, t.Dict]:
        """Apply an action to the environment.

        Args:
            action:

        Returns:
            A tuple containing:
                - A numpy ndarray containing the current environment state.
                - The reward obtained by applying the provided action.
                - A boolean flag indicating whether the episode has ended.
                - An empty info dict.
        """
        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        return self.state, reward / 10, done, done, {}

    def reset(self, seed=None, options=None) -> t.Tuple[Frame, t.Dict]:
        """Resets the environment.

        Returns:
            The initial state of the new environment.
        """
        self.game.new_episode()
        self.state = self._get_frame()

        return self.state, {}

    def close(self) -> None:
        self.game.close()

    def render(self, mode='human'):
        if (mode == "rgb_array"):
            return self.state

    def _get_frame(self, done: bool = False) -> Frame:
        state = self.game.get_state()
        if (state):
            return self.frame_processor(
                np.transpose(
                    state.screen_buffer, [1, 2, 0])) if not done else self.empty_frame
        else:
            return self.empty_frame


def create_env(scenario: str, visible = False, **kwargs) -> DoomEnv:
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/{scenario}.cfg')
    game.set_window_visible(visible)
    game.init()

    # Wrap the game with the Gym adapter.
    return DoomEnv(game, **kwargs)


def create_vec_env(n_envs=1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: create_env(**kwargs)] * n_envs))


def create_eval_vec_env(**kwargs) -> vec_env.VecTransposeImage:
    return create_vec_env(n_envs=1, **kwargs)
