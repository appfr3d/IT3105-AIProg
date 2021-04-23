import ConfigReader
from Actor import NNActor
from SimWorld import ShapeType
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm
import time
import numpy as np

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
      #self.test()
      #self.display_game()

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
    self.test()
    savepath = 'models/' + str(time.time())
    self.actor.model.save(savepath)
    self.display_log(savepath)
    self.config.copy_config(savepath)

  def run_episode(self):
    # Reset eligibility
    self.actor.reset_eligibility()

    # SARSA with eligibility traces w/nn functional approximator for Q
    old_state_actions = self.sim_world_player.get_state()
    old_action, old_state = self.actor.select_action(old_state_actions)
    win = False
    last_win = False

    while not last_win:
      self.sim_world_player.do_action(old_action)
      new_state_actions = self.sim_world_player.get_state()
      reinforcement = self.sim_world_player.get_reward()

      new_action, new_state = self.actor.select_action(new_state_actions)

      # Implemented in SplitGD by default
      TD = self.actor.get_TD_error(reinforcement, old_state, new_state)
      #print(self.actor.model(old_state), TD)
      self.actor.update(reinforcement, old_state, new_state)
      #print(self.actor.model(old_state))

      old_state = new_state
      old_action = new_action
      last_win = win
      win = self.sim_world_player.get_game_over()

    if self.sim_world_player.get_log_metric() < 1000:
      print(self.sim_world_player.get_log_metric())
    self.peg_log.append(self.sim_world_player.get_log_metric())
  
  def display_log(self, savepath):
    fig, ax = plt.subplots()
    ax.scatter(range(0, len(self.peg_log)), self.peg_log)
    plt.xlabel("Episode")
    plt.ylabel("Number of moves")
    
    plot_name = savepath + "/graph.png"
    plt.savefig(plot_name)
    plt.close(fig)
    plt.show()

  def display_game(self):
    self.actor.epsilon = 0
    self.sim_world_player.display = True
    self.sim_world_player.force_display_frame()

    self.run_episode()

  def test(self):
    test_tuples = []
    for num in range(-240, 120):
      for num2 in range(-14, 14):
        test_tuples.append((num/200, num2/200))

    nn_inputs = []
    for tuple in test_tuples:
      nn_inputs.append(self.sim_world_player.process_x_vel(tuple[0], tuple[1]))

    q_vals = []
    for inp in nn_inputs:
      part = []
      for inp2 in inp:
        part.append(self.actor.model(inp2).numpy()[0][0])
      q_vals.append(part)

    q_vals.reverse()
    print("")
    for num in range(self.config.tiles):
      for num2 in range(self.config.tiles):
        r = max(q_vals[num*self.config.tiles + num2])
        if q_vals[num*self.config.tiles + num2][0] == r:
          print("L", end='')
        if q_vals[num*self.config.tiles + num2][1] == r:
          print("N", end='')
        if q_vals[num*self.config.tiles + num2][2] == r:
          print("R", end='')
      print("")
