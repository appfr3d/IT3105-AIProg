import ConfigReader
import random

class Critic():
  """
  abstract
  """
  def eligibility_decay(self, state):
    pass
  
  def value_update(self, state, TD_error):
    pass

  def set_eligibility(self, state, value):
    pass
  
  def get_TD_error(self, old_state, new_state):
    pass

class TableCritic(Critic):
  def __init__(self, config: ConfigReader):
    self.learning_rate = config.critic_learning_rate
    self.eligibility_decay_rate = config.critic_eligibility_decay_rate
    self.discount_factor = config.critic_discount_factor
    self.state_value_map = {}
    self.state_eligibility_map = {}

  def check_and_initialize_value(self, state):
    """
    if the value is not in the state_value_map, initialize it wit a small random value
    """
    if not state in self.state_value_map:
      self.state_value_map[state] =  random.uniform(0, 0.1)
      self.state_eligibility_map[state] = 0
      
  def eligibility_decay(self, state):
    """
    Decays eligibility of a given state
    """
    self.state_eligibility_map[state] = self.state_eligibility_map[state]*self.discount_factor*self.eligibility_decay_rate
  
  def value_update(self, state, TD):
    """
    Updates the value of a given state using TD error
    """
    self.state_value_map[state] = self.state_value_map[state] + self.learning_rate*TD*self.state_eligibility_map[state]

  def set_eligibility(self, state, value):
    """
    Sets the eligibility of a given state
    """
    self.state_eligibility_map[state] = value
  
  def reset_eligibility(self):
    for key in self.state_eligibility_map.keys():
      self.state_eligibility_map[key] = 0
  
  def get_TD_error(self, reinforcement, old_state, new_state):
    """
    :returns the calculated TD error
    """
    # Note that it is only necessary to initialize the states here, because this function is always 
    # called first in the algorithm
    self.check_and_initialize_value(new_state) 
    self.check_and_initialize_value(old_state)
    
    return reinforcement + self.discount_factor*self.state_value_map[new_state] - self.state_value_map[old_state]