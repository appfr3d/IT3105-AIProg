import ConfigReader
from ActorNN import ActorNN, HexBoardNNBridge
from simWorld import ShapeType
from SimWorldDisplayer import ImageDisplay
from matplotlib import pyplot as plt

import math
import matplotlib
from tqdm import tqdm

from HexGameBoard import HexGameBoard

from PlayerEnum import Player
from MonteCarloTreeNodes import TreeNode, TreeState, HexGameBridge

from tensorflow.keras.callbacks import TensorBoard

class ReinforcementLearner():
  def __init__(self, config: ConfigReader, nn_bridge, game_bridge, model_path=None):
    """
    :param config: A config object containing configuration information.
    :param nn_bridge: game specific nn bridge
    :param game_bridge: game spesific rule bridge
    """

    self.actor = ActorNN(config, nn_bridge, model_path)
    self.config = config
    self.game_bridge = game_bridge

    
  def fit(self):
    # Run all episodes
    for episode in tqdm(range(self.config.number_of_episodes), desc="Episode"):
      self.run_episode(display=False)
      #if self.sim_world_player.get_reward() == 1.0:
      #  self.actor.epsilon_decay()

      # Multiplicative
      self.actor.epsilon *= self.config.epsilon_decay_rate
      #   self.actor.epsilon_decay()

      # If we are on save interval
      if (episode + 1) % int(math.floor(self.config.number_of_episodes / self.config.model_count)) == 0:
        self.actor.save(episode)

      # Logaritmic
      # if self.peg_log[-1] == 1:
      # self.actor.epsilon = self.config.initial_epsilon * 10**(-(2*episode)/self.config.number_of_episodes)

      # Linear
      # self.actor.epsilon = self.config.initial_epsilon * (1 - (episode/self.config.number_of_episodes))
      
    
    
  def run_episode(self, display=False):
    # 1: Make monte carlo tree
    # Get training samples
    # Do action 
    # If not end goto 1
    # If end do training, return

    game_state = self.game_bridge.initialize_new_state()
    if display:
      sim_world_displayer = ImageDisplay(game_state)
    tree_state = TreeState(self.game_bridge)
    root_node = TreeNode(self.config, game_state, Player.PLAYER1, tree_state, self.actor, self.actor.epsilon, self.game_bridge)
    RBUF = []

    while not self.game_bridge.get_win(root_node.state):
      RBUF_pair, next_root = root_node.monte_carlo_action()
      RBUF.append(RBUF_pair)

      root_node = next_root
      root_node.reset_parents() # Remove links to rest of tree so automatic memory management can do it's thing
      if display:
        sim_world_displayer.set_sim_world(root_node.state)
        sim_world_displayer.display(self.config.frame_delay)

    # Train ANET on a random minibatch of cases from RBUF:
    self.actor.fit(RBUF)

  def display_game(self):
    self.actor.epsilon = 0
    # self.sim_world_player.display = True
    # self.sim_world_player.force_display_frame()

    self.run_episode(display=True)
  
  def load(self, path):
    self.actor.load(path)

  def eval(self, data):
    return self.actor.eval(data)