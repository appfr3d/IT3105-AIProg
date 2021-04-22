import math
from BasicClientActorAbs import BasicClientActorAbs

import os
from ConfigReader import ConfigReader
from ActorNN import ActorNN, HexBoardNNBridgeOnlineTournament
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class BasicClientActor(BasicClientActorAbs):

    def __init__(self, IP_address=None, verbose=True):
        self.series_id = -1
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)

        config = ConfigReader()
        config.read_config('oht_config.txt')
        self.nn_bridge = HexBoardNNBridgeOnlineTournament(config)
        
        # Find the last model made
        oht_model_dir = os.path.join(CURRENT_DIR, 'tournament_models', 'oht_test')
        model_dirs = [d for d in os.listdir(oht_model_dir) if os.path.isdir(os.path.join(oht_model_dir, d))]
        model_path = os.path.join(oht_model_dir, model_dirs[-2])
        print(model_path)

        self.actor = ActorNN(config, self.nn_bridge, model_path)

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """
        next_move = self.actor.eval(state)
        return next_move

    # The rest of the stuff seeems fine as it was  
    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return
        """
        print('unique_id', unique_id)
        print('series_id', series_id)
        print('player_map', player_map)
        print('num_games', num_games)
        print('game_params', game_params)

        self.nn_bridge.handle_series_start(series_id, player_map)
        self.series_id = series_id

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        print('start_player', start_player)

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))
        self.nn_bridge.handle_game_over(winner)

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    bsa = BasicClientActor(verbose=False)
    bsa.connect_to_server()

# hard model before first-random-greedy
'''
Game start
2
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 2, 2, 1, 0, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
Game end
Game over, these are the stats:
Winner: 2
End state: (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 0, 0, 2, 2, 1, 0, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
Game start
1
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 2, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 2, 1, 0, 0, 2, 2, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
Game end
Game over, these are the stats:
Winner: 2
End state: (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 0, 0, 1, 2, 1, 0, 0, 2, 2, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
Series end
Series ended, these are the stats:
[(6371936, 1, 0, 50), (2020, 2, 50, 0)]
Tournament end
Tournament over. Your score was: 67.0
'''

'''
Winner: 2
(1, 0, 0, 2, 2, 2, 2, 0, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 0, 0, 0, 2, 0, 2, 0)
(1, 0, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0)
(1, 0, 2, 2, 0, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2)
(1, 0, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 0, 1, 1, 2, 2, 2, 2)
(1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 0, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2)
(1, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1, 2, 0, 2, 2, 1, 2, 2, 2)

(1, 2, 0, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 0, 2)
(1, 2, 0, 2, 0, 1, 1, 2, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 2, 2, 2, 2, 2, 2)
(1, 2, 0, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 0, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0)
(1, 2, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 0, 1, 2, 1, 0, 2, 2)
(1, 2, 2, 1, 2, 2, 1, 0, 2, 1, 1, 1, 2, 0, 1, 2, 1, 1, 1, 0, 2, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 2, 2, 2, 0, 0, 0)
(1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2)
'''