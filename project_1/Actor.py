import ConfigReader
import random
import math

class Actor:
  def __init__(self, config: ConfigReader):
    self.eligibility_decay_rate = config.actor_eligibility_decay_rate
    self.discount_factor = config.actor_discount_factor
    self.learning_rate = config.actor_learning_rate
    self.epsilon = config.initial_epsilon
    self.config = config
    self.state_value_map = {}
    self.state_eligibility_map = {}

  def select_action(self, state, actions):
    # A state can have no other actions than all actions
    if not state in self.state_value_map: 
      action_map = {}
      eligibility_map = {}
      for a in actions:
        action_map[a] = 0
        eligibility_map[a] = 0
      self.state_value_map[state] = action_map
      self.state_eligibility_map[state] = eligibility_map
    
    # Decide whether to explore or to exploit
    random.seed()
    rand = random.uniform(0, 1)
    if rand < self.epsilon:
      # explore, choose a random action
      key_options = list(self.state_value_map[state].keys())
      action_key = random.choice(key_options)
    else:
      # exploit, choose the action with the highest value
      key_list = list(self.state_value_map[state].keys())
      action_key = key_list[0]
      for action in key_list[1:]:
        if self.state_value_map[state][action] > self.state_value_map[state][action_key]:
          action_key = action
          
    return action_key

  def set_eligibility(self, state, action, value):
    self.state_eligibility_map[state][action] = value
  
  def eligibility_decay(self, state, action):
    self.state_eligibility_map[state][action] = self.state_eligibility_map[state][action]*self.discount_factor*self.eligibility_decay_rate
  
  def value_update(self, state, action, TD):
    self.state_value_map[state][action] = self.state_value_map[state][action] + self.learning_rate*TD*self.state_eligibility_map[state][action]
  
  def epsilon_decay(self):
    """
    Decays epsilon slightly. Great performance improvement
    """
    self.epsilon = max(self.config.initial_epsilon/10.0, self.epsilon * 0.9)

  def reset_eligibility(self):
    for key1 in self.state_eligibility_map.keys():
      for key2 in self.state_eligibility_map[key1].keys():
        self.state_eligibility_map[key1][key2] = 0

