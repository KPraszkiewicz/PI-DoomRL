from vizdoom.vizdoom import GameVariable
from DoomEnv import DoomEnv

class DoomWithBots(DoomEnv):

    def __init__(self, game, frame_processor, frame_skip, n_bots, combinated_buttons = True):
        super().__init__(game, frame_processor, frame_skip, combinated_buttons)
        self.n_bots = n_bots
        self.last_frags = 0    
        self._reset_bots()
        

    def step(self, action):
        self.game.make_action(self.possible_actions[action], self.frame_skip)
       
        # Compute rewards.
        frags = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        reward = frags - self.last_frags
        self.last_frags = frags

        # Check for episode end.
        self._respawn_if_dead()
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        return self.state, reward, done, done, {}

    def reset(self, seed=None, options=None):
        self._reset_bots()
        self.last_frags = 0

        return super().reset(seed, options)

    def _respawn_if_dead(self):
        if not self.game.is_episode_finished():
            if self.game.is_player_dead():
                self.game.respawn_player()
                
    def _reset_bots(self):
        # Make sure you have the bots.cfg file next to the program entry point.
        self.game.send_game_command('removebots')
        for i in range(self.n_bots):
            self.game.send_game_command('addbot')