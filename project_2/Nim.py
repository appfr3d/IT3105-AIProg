# For testing the MCTS
from MonteCarloTreeNodes import GameBridge, TreeNode, TreeState
from PlayerEnum import Player
import ConfigReader
import random
import tqdm
import matplotlib.pyplot as plt

global_n = 25
global_k = 10
# This is a bit quick and hacky, but it is just intended to test MonteCarloTreeNodes
class Nim():
    def __init__(self, N, K, turn = 0):
        self.num = N
        self.max_sub = K
        self.turn = turn

    def act(self, sub):
        self.num -= sub
        if self.turn == 0:
            self.turn = 1
        else:
            self.turn = 0

    def get_all_moves(self):
        if self.max_sub < self.num:
            return [num for num in range(1, self.max_sub+1)]
        else:
            return [n for n in range(1, self.num+1)]

    def get_win(self):
        if self.num == 0:
            if self.turn == 0:
                return 1 # The player who has the losing hand is the loser
            if self.turn == 1:
                return 0
        else:
            return "FalseToken"


class NimGameBridge(GameBridge):
    def __init__(self, config, N, K):
        super().__init__(config)
        self.N = N
        self.K = K

    def initialize_new_state(self):
        return Nim(self.N, self.K)

    def get_state(self, state):
        return (state.num, state.turn)

    def get_max_possible_actions(self):
        return self.K

    def get_dist_index(self, move):
        return move[0]-1

    def get_winner_data(self, state):
        winner = state.get_win()
        if winner == 0:
            to_return = 1
        elif winner == 1:
            to_return = -1
        else:
            # should not happen but for debugging
            raise Exception("No more moves but no winner yet, critical game logic failure")
        return to_return

    def get_move_count(self, state):
        return len(state.get_all_moves())

    def get_all_tree_moves(self, state):
        return state.get_all_moves()

    def execute_move(self, state, move):
        new_state = Nim(state.num, state.max_sub, state.turn)
        new_state.act(move[0])
        return new_state

    def get_win(self, state):
        return state.get_win() != "FalseToken"

    def hash(self, state):
        return (state.num, state.turn)


config100 = ConfigReader.ConfigReader()
config1000 = ConfigReader.ConfigReader()
config100.rollouts_per_move = 100
config1000.rollouts_per_move = 1000

class NimPlayer():
    def __init__(self, config = "", random=False):
        self.config = config
        self.random = random

    def eval(self, state):
        if self.random:
            return random.choice(state.get_all_moves())
        else:
            game_bridge = NimGameBridge(self.config, global_n, global_k)
            tree_state = TreeState(game_bridge)
            epsilon = 1
            default_policy = lambda x: random.choice(x)
            if state.turn == 0:
                player = Player.PLAYER1
            else:
                player = Player.PLAYER2
            node = TreeNode(self.config, state, player, tree_state, default_policy, epsilon, game_bridge)
            rbuf = node.monte_carlo_action()[0]
            dist = rbuf[1]
            return list(dist).index(max(dist))+1


nim100 = NimPlayer(config=config100)
nim1000 = NimPlayer(config=config1000)
nim_random = NimPlayer(random=True)


class NimTournament:
    def __init__(self, agent_dict):
        self.agent_dict = agent_dict
        self.game_bridge = NimGameBridge("", global_n, global_k)

    def run_tourney(self):
        games_per_series = 10
        agent_score_dict = {}

        # Initialize agent_score_dicts
        for name in self.agent_dict:
            agent_score_dict[name] = {}
            for name2 in self.agent_dict:
                agent_score_dict[name][name2] = 0

        # For each pair run a tournament
        for name in tqdm.tqdm(self.agent_dict, desc="Names"):
            for name2 in self.agent_dict:
                if agent_score_dict[name][name2] != 0 or agent_score_dict[name2][name] != 0:
                    # If already played don't play again
                    continue
                elif name == name2:
                    # No need to play against itself
                    continue
                else:
                    player1wins = 0
                    player2wins = 0
                    for num in range(games_per_series // 2):
                        winner = self.run_game(self.agent_dict[name], self.agent_dict[name2])
                        if winner == Player.PLAYER1:
                            player1wins += 1
                        else:
                            player2wins += 1
                    # Play with opposite player1/player2
                    for num in range(games_per_series // 2):
                        winner = self.run_game(self.agent_dict[name2], self.agent_dict[name])
                        if winner == Player.PLAYER1:
                            player2wins += 1
                        else:
                            player1wins += 1
                    agent_score_dict[name][name2] = player1wins
                    agent_score_dict[name2][name] = player2wins

        win_count_list = []
        # Print results
        for name in self.agent_dict:
            print("========================================================")
            print("Results for agent " + name)
            print("--------------------------------------------------------")

            total_wins = 0
            for name2 in self.agent_dict:
                if name != name2:
                    print("Result against " + name2 + " is: " + str(agent_score_dict[name][name2]) + " wins")
                    total_wins += agent_score_dict[name][name2]
            print('Total number of wins: ' + str(total_wins))
            print("\n\n\n")
            win_count_list.append(total_wins)
        plt.bar(['random', '100', '1000'], win_count_list)
        plt.show()

    def run_game(self, agent1, agent2):
        """
        init hex board

        while not finished:
          get board state, possible moves, player to move
          run actor_nn eval on player to move
          do action

        return winner
        """
        player_to_move = Player.PLAYER1
        state = self.game_bridge.initialize_new_state()
        while not self.game_bridge.get_win(state):
            if player_to_move == Player.PLAYER1:
                # player1
                action = agent1.eval(state)
                state = self.game_bridge.execute_move(state, (action, Player.PLAYER1))
                player_to_move = Player.PLAYER2
            else:
                # player2
                moves = self.game_bridge.get_all_nn_moves(state)
                action = agent2.eval(state)
                state = self.game_bridge.execute_move(state, (action, Player.PLAYER2))
                player_to_move = Player.PLAYER1
        winner = self.game_bridge.get_winner_data(state)
        if winner == 1:
            return Player.PLAYER1
        return Player.PLAYER2

agent_dict = {'random':nim_random, '100':nim100, '1000':nim1000}

tourney = NimTournament(agent_dict)
tourney.run_tourney()