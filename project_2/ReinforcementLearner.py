import ConfigReader
from ActorNN import ActorNN, HexBoardNNBridge
from simWorld import ShapeType
from SimWorldDisplayer import ImageDisplay
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm

from HexGameBoard import HexGameBoard

from PlayerEnum import Player
from MonteCarloTreeNodes import TreeNode, TreeState

from tensorflow.keras.callbacks import TensorBoard

class ReinforcementLearner():
  def __init__(self, config: ConfigReader):
    """
    :param actor: An actor agent
    :param config: A config object containing configuration information.
    """
    self.actor = ActorNN(config, HexBoardNNBridge(config))
    self.config = config

    
  def fit(self):
    # Run all episodes
    for episode in tqdm(range(self.config.number_of_episodes), desc="Episode"):
      self.run_episode(display=episode%5==0)
      #if self.sim_world_player.get_reward() == 1.0:
      #  self.actor.epsilon_decay()

      # Multiplicative
      self.actor.epsilon *= self.config.epsilon_decay_rate
      #   self.actor.epsilon_decay()
      
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

    # REFACTOR
    board = HexGameBoard(self.config.board_type, self.config.size)
    if display:
      sim_world_displayer = ImageDisplay(board)
    tree_state = TreeState()
    root_node = TreeNode(self.config, board, Player.PLAYER1, tree_state, self.actor, self.actor.epsilon)
    RBUF = []

    while not root_node.hex_board.get_win():
      RBUF_pair, next_root = root_node.monte_carlo_action()
      RBUF.append(RBUF_pair)

      root_node = next_root
      root_node.reset_parents() # Remove links to rest of tree so automatic memory management can do it's thing
      if display:
        sim_world_displayer.set_sim_world(root_node.hex_board)
        sim_world_displayer.display(self.config.frame_delay)

    # Train ANET on a random minibatch of cases from RBUF:
    self.actor.fit(RBUF)
  
  # def display_log(self):
  #   plt.plot(self.peg_log)
  #   plt.xlabel("Episode")
  #   plt.ylabel("Pegs remaining")

  #   plot_name = "graphs/graph.png"
  #   plt.savefig(plot_name)
  #   plt.show()

  def display_game(self):
    self.actor.epsilon = 0
    # self.sim_world_player.display = True
    # self.sim_world_player.force_display_frame()

    self.run_episode(display=True)