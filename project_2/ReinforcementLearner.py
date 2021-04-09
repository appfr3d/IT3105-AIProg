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
import cProfile
import os
import random

class RBUF_OBJECT():
  def __init__(self, max_index):
    self.list = []
    self.index = 0
    self.max_index = max_index

  def append(self, list_object):
    if len(self.list) == self.max_index+1:
      self.list[self.index] = list_object
      self.index = (self.index + 1) % self.max_index
    else:
      self.list.append(list_object)

  def get_minibatch(self, minibatch_size=32):
    if len(self.list) < minibatch_size:
      return self.list
    else:
      return random.choices(population=self.list, k=32)

class ReinforcementLearner():
  def __init__(self, config: ConfigReader, nn_bridge, game_bridge, save_path=None, model_path=None):
    """
    :param config: A config object containing configuration information.
    :param nn_bridge: game specific nn bridge
    :param game_bridge: game spesific rule bridge
    """

    self.actor = ActorNN(config, nn_bridge, model_path)
    self.config = config
    self.game_bridge = game_bridge
    self.save_path = save_path
    self.model_path = model_path

    
  def fit(self):
    # Run all episodes
    RBUF = RBUF_OBJECT(self.config.rbuf_size)

    # if there are some directories in self.save_path
    # start from the highest number
    if not self.model_path == None:
      # find the episode number from the model_path
      _, tail = os.path.split(self.model_path)
      start_episode = int(tail)

      # Calculate the last epsilon value and set initial epsilon to that value
      e = self.config.initial_epsilon
      for _ in range(start_episode):
        e *= self.config.epsilon_decay_rate
      self.actor.epsilon = max(e, self.config.epsilon_lower_bound)

      print('Starting traingin on episode ' + str(start_episode))

    else:
      start_episode = 0

    for episode in tqdm(range(start_episode, start_episode + self.config.number_of_episodes), desc="Episode"):
      RBUF = self.run_episode(RBUF)
      #if self.sim_world_player.get_reward() == 1.0:
      #  self.actor.epsilon_decay()

      # Multiplicative
      self.actor.epsilon = max(self.actor.epsilon*self.config.epsilon_decay_rate, self.config.epsilon_lower_bound) 
      # self.actor.epsilon_decay()

      # If we are on save interval
      if (episode + 1) % int(math.floor(self.config.number_of_episodes / self.config.model_count)) == 0:
        self.actor.save(self.save_path, (episode + 1))

      # Logaritmic
      # if self.peg_log[-1] == 1:
      # self.actor.epsilon = self.config.initial_epsilon * 10**(-(2*episode)/self.config.number_of_episodes)

      # Linear
      # self.actor.epsilon = self.config.initial_epsilon * (1 - (episode/self.config.number_of_episodes))
    # TODO: If last episode is not saved, save it here...
    # Read the directories from self.save_path if it is not None and check if the last episode is in it.
    
    
  def run_episode(self, RBUF):
    # 1: Make monte carlo tree
    # Get training samples
    # Do action 
    # If not end goto 1
    # If end do training, return

    game_state = self.game_bridge.initialize_new_state()
    tree_state = TreeState(self.game_bridge)
    root_node = TreeNode(self.config, game_state, Player.PLAYER1, tree_state, self.actor, self.actor.epsilon, self.game_bridge)
    while not self.game_bridge.get_win(root_node.state):
      RBUF_pair, next_root = root_node.monte_carlo_action()
      RBUF.append(RBUF_pair)

      root_node = next_root
      root_node.reset_parents() # Remove links to rest of tree so automatic memory management can do it's thing
      if self.config.display:
        sim_world_displayer = ImageDisplay(root_node.state)
        sim_world_displayer.display(self.config.frame_delay)

    # Train ANET on a random minibatch of cases from RBUF:
    self.actor.fit(RBUF.list)
    return RBUF

  def display_game(self):
    self.actor.epsilon = 0
    # self.sim_world_player.display = True
    # self.sim_world_player.force_display_frame()
    RBUF = RBUF_OBJECT(200)
    self.run_episode(RBUF)
  
  def load(self, path):
    self.actor.load(path)

  def eval(self, data):
    return self.actor.eval(data)