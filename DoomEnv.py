import typing as t

import numpy as np
import vizdoom
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from stable_baselines3.common import vec_env
from vizdoom import Button
import itertools

Frame = np.ndarray

# Buttons that cannot be used together
MUTUALLY_EXCLUSIVE_GROUPS = [
    [Button.MOVE_RIGHT, Button.MOVE_LEFT],
    [Button.TURN_RIGHT, Button.TURN_LEFT],
    [Button.MOVE_FORWARD, Button.MOVE_BACKWARD],
]

# Buttons that can only be used alone.
EXCLUSIVE_BUTTONS = [Button.ATTACK]

class DoomEnv(Env):
    """Wrapper environment following OpenAI's gym interface for a VizDoom game instance."""

    def __init__(self,
                 game: vizdoom.DoomGame,
                 frame_processor: t.Callable,
                 frame_skip: int = 4,
                 compress_buttons: bool = True):
        super().__init__()

    
        # Determine observation space
        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        new_h, new_w, new_c = frame_processor(np.zeros((h, w, c))).shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8)

        # Assign other variables
        self.game = game

        if compress_buttons:
            self.possible_actions = get_available_actions(game.get_available_buttons())
            self.action_space = spaces.Discrete(len(self.possible_actions))
        else:
            self.action_space = spaces.Discrete(game.get_available_buttons_size())
            self.possible_actions = np.eye(self.action_space.n).tolist()  # VizDoom needs a list of buttons states.
        
        # Determine action space
        

        print(self.possible_actions)
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

        return self.state, reward, done, done, {}

    def reset(self, seed=None, options=None) -> t.Tuple[Frame, t.Dict]:
        """Resets the environment.

        Returns:
            The initial state of the new environment.
        """
        self.game.new_episode()
        if seed:
            self.game.set_seed(seed)
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

# ENVS

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

# BUTTONS

def has_exclusive_button(actions: np.ndarray, buttons: np.array) -> np.array:
    exclusion_mask = np.isin(buttons, EXCLUSIVE_BUTTONS)
    
    # Flag actions that have more than 1 active button among exclusive list.
    return (np.any(actions.astype(bool) & exclusion_mask, axis=-1)) & (np.sum(actions, axis=-1) > 1)

def has_excluded_pair(actions: np.ndarray, buttons: np.array) -> np.array:
    # Create mask of shape (n_mutual_exclusion_groups, n_available_buttons), marking location of excluded pairs.
    mutual_exclusion_mask = np.array([np.isin(buttons, excluded_group) 
                                      for excluded_group in MUTUALLY_EXCLUSIVE_GROUPS])

    # Flag actions that have more than 1 button active in any of the mutual exclusion groups.
    return np.any(np.sum(
        # Resulting shape (n_actions, n_mutual_exclusion_groups, n_available_buttons)
        (actions[:, np.newaxis, :] * mutual_exclusion_mask.astype(int)),
        axis=-1) > 1, axis=-1)

def get_available_actions(buttons: np.array) -> t.List[t.List[float]]:
    # Create list of all possible actions of size (2^n_available_buttons x n_available_buttons)
    action_combinations = np.array([list(seq) for seq in itertools.product([0., 1.], repeat=len(buttons))])

    # Build action mask from action combinations and exclusion mask
    illegal_mask = (has_excluded_pair(action_combinations, buttons)
                    | has_exclusive_button(action_combinations, buttons))

    possible_actions = action_combinations[~illegal_mask]
    possible_actions = possible_actions[np.sum(possible_actions, axis=1) > 0]  # Remove no-op

    print(f'Built action space of size {len(possible_actions)} from buttons {buttons}')
    return possible_actions.tolist()