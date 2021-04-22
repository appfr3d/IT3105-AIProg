import ConfigReader
from Actor import NNActor
from SimWorld import ShapeType
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm
import time

from tensorflow.keras.callbacks import TensorBoard

class ReinforcementLearner():
  def __init__(self, sim_world_player, config: ConfigReader):
    """
    :param actor: An actor agent
    :param critic: A critic agent
    :param config: A config object containing configuration information.
    """

    # Init critic based in config.critic_type
    self.actor = NNActor(config)

    self.config = config
    self.sim_world_player = sim_world_player
    self.state_action_pair_list = []
    self.peg_log = []  # Log of how many remaning pegs there were at the end of an epsiode

    
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
      
    savepath = 'models/' + str(time.time())
    self.actor.model.save(savepath)
    self.display_log(savepath)
    
  def run_episode(self):
    # Reset eligibility
    self.actor.reset_eligibility() 

    # SARSA with eligibility traces w/nn functional approximator for Q 
    old_state_actions = self.sim_world_player.get_state()
    old_action, old_state = self.actor.select_action(old_state_actions)
    
    while not self.sim_world_player.get_game_over():
      self.sim_world_player.do_action(old_action)
      new_state_actions = self.sim_world_player.get_state()
      reinforcement = self.sim_world_player.get_reward()
      
      new_action, new_state = self.actor.select_action(new_state_actions)
      
      # Implemented in SplitGD by default
      # TD = self.actor.get_TD_error(reinforcement, old_state, new_state)

      self.actor.update(reinforcement, old_state, new_state)

      old_state = new_state
      old_action = new_action

    self.peg_log.append(self.sim_world_player.get_log_metric())
  
  def display_log(self, savepath):
    plt.scatter(range(0, len(self.peg_log)), self.peg_log)
    plt.xlabel("Episode")
    plt.ylabel("Number of moves")
    
    plot_name = savepath + "/graph.png"
    plt.savefig(plot_name)
    plt.show()

  def display_game(self):
    self.actor.epsilon = 0
    self.sim_world_player.display = True
    self.sim_world_player.force_display_frame()

    self.run_episode()
