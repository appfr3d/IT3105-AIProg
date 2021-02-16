from PegSolitairePlayer import PegSolitairePlayer
import ConfigReader
from Actor import Actor
from Critic import TableCritic, NNCritic, CriticType
from simWorld import ShapeType
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm

from tensorflow.keras.callbacks import TensorBoard

class ReinforcementLearner():
  def __init__(self, sim_world_player, config: ConfigReader):
    """
    :param actor: An actor agent
    :param critic: A critic agent
    :param config: A config object containing configuration information.
    """
    self.actor = Actor(config)

    # Init critic based in config.critic_type
    if config.critic_type == CriticType.TABLE:
      self.critic = TableCritic(config) 
    elif config.critic_type == CriticType.NN:
      self.critic = NNCritic(config)

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
      
    
    self.display_log()
    
  def run_episode(self):
    # Based on 3.1 generic Actor-Critic algorithm
    # Reset eligibility
    self.actor.reset_eligibility() 
    self.critic.reset_eligibility()

    if isinstance(self.critic, TableCritic):
      # We are just updating actions used in this episode
      self.state_action_pair_list = []  

      old_state = self.sim_world_player.get_state()
      possible_actions = self.sim_world_player.get_actions()
      old_action = self.actor.select_action(old_state, possible_actions)
      
      while not self.sim_world_player.get_game_over():
        self.sim_world_player.do_action(old_action)
        new_state = self.sim_world_player.get_state()
        reinforcement = self.sim_world_player.get_reward()
        
        if not self.sim_world_player.get_game_over():
          # When there are no more remaining moves skip preparing next move
          new_action = self.actor.select_action(new_state, self.sim_world_player.get_actions())

        self.actor.set_eligibility(old_state, old_action, 1)
        
        self.state_action_pair_list.append((old_state, old_action))

        TD = self.critic.get_TD_error(reinforcement, old_state, new_state)
        self.critic.set_eligibility(old_state, 1)

        for (s, a) in self.state_action_pair_list:
          self.critic.value_update(s, TD)
          self.critic.eligibility_decay(s)
          
          self.actor.value_update(s, a, TD)
          self.actor.eligibility_decay(s, a)
        
        old_state = new_state
        old_action = new_action
    elif isinstance(self.critic, NNCritic):
      # We are just updating actions used in this episode
      self.state_action_pair_list = []  

      old_state = self.sim_world_player.get_state()
      possible_actions = self.sim_world_player.get_actions()
      old_action = self.actor.select_action(old_state, possible_actions)
      
      while not self.sim_world_player.get_game_over():
        self.sim_world_player.do_action(old_action)
        new_state = self.sim_world_player.get_state()
        reinforcement = self.sim_world_player.get_reward()
        
        if not self.sim_world_player.get_game_over():
          # When there are no more remaining moves skip preparing next move
          new_action = self.actor.select_action(new_state, self.sim_world_player.get_actions())

        self.actor.set_eligibility(old_state, old_action, 1)
        
        self.state_action_pair_list.append((old_state, old_action))

        TD = self.critic.get_TD_error(reinforcement, old_state, new_state)

        for (s, a) in self.state_action_pair_list:
          self.critic.value_update(s, TD)
          
          self.actor.value_update(s, a, TD)
          self.actor.eligibility_decay(s, a)
        self.critic.update(reinforcement, old_state, new_state)

        old_state = new_state
        old_action = new_action

    self.peg_log.append(self.sim_world_player.get_remaining_pegs())
  
  def display_log(self):
    plt.plot(self.peg_log)
    plt.xlabel("Episode")
    plt.ylabel("Pegs remaining")
    
    if isinstance(self.critic, NNCritic):
      critic_name = "NN"
    else:
      critic_name = "table"
    
    if self.config.board_type == ShapeType.TRIANGLE:
      board_name = "triangle"
    else: 
      board_name = "diamond"
    plot_name = "graphs/graph_" + board_name + "_" + critic_name + ".png"
    plt.savefig(plot_name)
    plt.show()

  def display_game(self):
    self.actor.epsilon = 0
    self.sim_world_player.display = True
    self.sim_world_player.force_display_frame()

    self.run_episode()
