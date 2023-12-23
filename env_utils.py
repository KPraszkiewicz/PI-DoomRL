import vizdoom
from stable_baselines3.common import vec_env
from DoomEnv import DoomEnv

def create_env(scenario: str, visible = False, EnvClass = DoomEnv, **kwargs):
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/{scenario}.cfg')
    game.set_window_visible(visible)
    game.init()

    # Wrap the game with the Gym adapter.
    return EnvClass(game, **kwargs)


def create_vec_env(n_envs=1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: create_env(**kwargs)] * n_envs))


def env_with_bots(scenario, visible = False, EnvClass = DoomEnv, **kwargs):
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/{scenario}.cfg')
    game.add_game_args('-host 1 -deathmatch +viz_nocheat 0 +cl_run 1 +name AGENT +colorset 0' +
                       '+sv_forcerespawn 1 +sv_respawnprotect 1 +sv_nocrouch 1 +sv_noexit 1')      # Players can't crouch.
    game.set_window_visible(visible)
    game.init()

    return EnvClass(game, **kwargs)

def vec_env_with_bots(n_envs=1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: env_with_bots(**kwargs)] * n_envs))