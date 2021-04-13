import os
import random

from tqdm import tqdm
from ReinforcementLearner import ReinforcementLearner
from ActorNN import HexBoardNNBridge
from MonteCarloTreeNodes import HexGameBridge
from PlayerEnum import Player
from SimWorldDisplayer import ImageDisplay
import matplotlib.pyplot as plt
class Tournament:
  def __init__(self, config, model_save_path, game_bridge, nn_bridge):
    self.config = config
    self.model_save_path = model_save_path
    self.game_bridge = game_bridge
    self.nn_bridge = nn_bridge

    # When using epsilon greedy evaluation we use a little bit of variation, just so that a otherwise "good" model
    # does not get stuck in a "bad" track that is deterministic from the start state.
    self.config.initial_epsilon = 0.1

  def run_tourney(self):
    games_per_series = self.config.games_per_series
    agent_names = [a for a in os.listdir(self.model_save_path) if os.path.isdir(os.path.join(self.model_save_path, a))]
    agent_dict = {"Randy Random":"Randy Random"}
    agent_score_dict = {}

    # Make a dictionary for model names - agent
    for name in agent_names:
      rl = ReinforcementLearner(self.config, self.nn_bridge, self.game_bridge, self.model_save_path, self.model_save_path + '/' + name)
      agent_dict[name] = rl

    agent_names.append("Randy Random")

    
    # Initialize agent_score_dicts
    for name in agent_names:
      agent_score_dict[name] = {}
      for name2 in agent_names: 
        agent_score_dict[name][name2] = 0

    # For each pair run a tournament
    for name in tqdm(agent_names, desc="Names"):
      for name2 in agent_names:
        
        if agent_score_dict[name][name2] != 0 or agent_score_dict[name2][name] != 0:
          # If already played don't play again
          continue
        elif name == name2:
          # No need to play against itself
          continue
        else:
          player1wins = 0
          player2wins = 0
          for num in range(games_per_series//2):
            winner = self.run_game(agent_dict[name], agent_dict[name2])
            if winner == Player.PLAYER1:
              player1wins += 1
            else:
              player2wins += 1
          # Play with opposite player1/player2
          for num in range(games_per_series//2):
            winner = self.run_game(agent_dict[name2], agent_dict[name])
            if winner == Player.PLAYER1:
              player2wins += 1
            else:
              player1wins += 1
          agent_score_dict[name][name2] = player1wins
          agent_score_dict[name2][name] = player2wins

    win_count_list = []
    # Print results
    # Print results for each model
    total_wins_dict = {}
    for name in agent_names:
      print("========================================================")
      print("Results for agent " + name)
      print("--------------------------------------------------------")
      
      total_wins = 0
      for name2 in agent_names:
        if name != name2:
          print("Result against " + name2 + " is: " + str(agent_score_dict[name][name2]) + " wins")
          total_wins += agent_score_dict[name][name2]
      total_wins_dict[name] = total_wins
      print('Total number of wins: ' + str(total_wins))
      print("\n\n\n")
      win_count_list.append(total_wins)
    fig, ax = plt.subplots()
    ax.bar(agent_names, win_count_list)
    #plt.show()
    fig.savefig(self.model_save_path + "/" + self.config.tournament_action_mode + ".png")

    # Print combined results for total wins
    print('Results   : | ', end='')
    for name in agent_names:
      print(name, end=' | ')
    print('\nTotal wins: | ', end='')
    for name in agent_names:
      print("{:>8}".format(total_wins_dict[name]), end=' | ')

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
    action_mode = self.config.tournament_action_mode
    while not self.game_bridge.get_win(state):
      if self.config.display:
        sim_world_displayer = ImageDisplay(state)
        sim_world_displayer.display(self.config.frame_delay)
      if player_to_move == Player.PLAYER1:
        # player1
        if agent1 == "Randy Random":
          moves = self.game_bridge.get_all_tree_moves(state)
          action = (random.choice(moves), player_to_move)
        else:
          moves = self.game_bridge.get_all_nn_moves(state)
          action = agent1.eval((moves, player_to_move, self.game_bridge.get_state(state), action_mode))
        state = self.game_bridge.execute_move(state, action)
        player_to_move = Player.PLAYER2
      else: 
        # player2
        if agent2 == "Randy Random":
          moves = self.game_bridge.get_all_tree_moves(state)
          action = (random.choice(moves), player_to_move)
        else:
          moves = self.game_bridge.get_all_nn_moves(state)
          action = agent2.eval((moves, player_to_move, self.game_bridge.get_state(state), action_mode))
        state = self.game_bridge.execute_move(state, action)
        player_to_move = Player.PLAYER1
    winner = self.game_bridge.get_winner_data(state)
    if self.config.display:
      sim_world_displayer = ImageDisplay(state)
      sim_world_displayer.display(self.config.frame_delay)
    if winner == 1:
      return Player.PLAYER1
    return Player.PLAYER2



