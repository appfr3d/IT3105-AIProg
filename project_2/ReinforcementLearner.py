from PegSolitairePlayer import PegSolitairePlayer
import ConfigReader
from ActorNN import ActorNN
from SimWorld import ShapeType
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm

from tensorflow.keras.callbacks import TensorBoard

class ReinforcementLearner():
  def __init__(self, sim_world_player, config: ConfigReader):
    """
    :param actor: An actor agent
    :param config: A config object containing configuration information.
    """
    self.actor = ActorNN(config)

    self.config = config
    self.sim_world_player = sim_world_player

    
  def fit(self):
    # Run all episodes
    for episode in tqdm(range(self.config.number_of_episodes), desc="Episode"):
      self.run_episode()
      #if self.sim_world_player.get_reward() == 1.0:
      #  self.actor.epsilon_decay()
      self.sim_world_player.reset_state()

      # Multiplicative
      self.actor.epsilon *= self.config.epsilon_decay_rate
      #   self.actor.epsilon_decay()
      
      # Logaritmic
      # if self.peg_log[-1] == 1:
      # self.actor.epsilon = self.config.initial_epsilon * 10**(-(2*episode)/self.config.number_of_episodes)

      # Linear
      # self.actor.epsilon = self.config.initial_epsilon * (1 - (episode/self.config.number_of_episodes))
      
    
    self.display_log()
    
  def run_episode(self):
    
    # 1: Make monte carlo tree
    # Get training samples
    # Do action 
    # If not end goto 1
    # If end do training, return
  
  def display_log(self):
    plt.plot(self.peg_log)
    plt.xlabel("Episode")
    plt.ylabel("Pegs remaining")

    plot_name = "graphs/graph.png"
    plt.savefig(plot_name)
    plt.show()

  def display_game(self):
    self.actor.epsilon = 0
    self.sim_world_player.display = True
    self.sim_world_player.force_display_frame()

    self.run_episode()