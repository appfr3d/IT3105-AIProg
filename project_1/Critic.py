import random
from enum import Enum
from tensorflow import keras
import tensorflow as tf

import ConfigReader
import SplitGD

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


class CriticType(Enum):
  TABLE = 1
  NN = 2

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
      self.state_value_map[state] = random.uniform(0, 0.1)
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


class NNCritic(Critic):
  def __init__(self, config: ConfigReader):
    self.learning_rate = config.critic_learning_rate
    self.eligibility_decay_rate = config.critic_eligibility_decay_rate
    self.discount_factor = config.critic_discount_factor
    self.nn_dimentions = config.critic_nn_dimentions
    self.input_size = (1, config.size*config.size)

    self.model = self.generate_fully_connected()
    self.nn = SplitGD.ReinforcementGD(self.model, self.discount_factor, self.eligibility_decay_rate)

    print(self.model.summary())

  def reset_eligibility(self):
    self.nn.eligibility_gradients = None
  
  def generate_conv(self):
    opt = keras.optimizers.SGD
    loss = keras.losses.MSE  # AS done in actor critic PDF
    model = keras.models.Sequential()

    # Add a Conv1D layer for each of the first specified dimentions
    for dim in self.nn_dimentions[:-1]:
      model.add(keras.layers.Conv1D(dim, kernel_size=3, strides=1, activation='relu'))
      model.add(keras.layers.MaxPooling1D(pool_size=3, strides=3))
    
    # Add the last layer with softmax as the activation function
    model.add(keras.layers.Dense(self.nn_dimentions[-1], activation='softmax')) 

    # Compile and return the model
    model.compile(optimizer=opt(lr=self.learning_rate), loss=loss, metrics=[keras.metrics.categorical_accuracy])
    model.build(input_shape = self.input_size)
    return model
  
  def generate_fully_connected(self):
    opt = keras.optimizers.SGD
    loss = keras.losses.MSE
    model = keras.models.Sequential()

    for dim in self.nn_dimentions[:-1]:
      model.add(keras.layers.Dense(dim, activation='relu'))
    
    model.add(keras.layers.Dense(self.nn_dimentions[-1], activation='relu'))

    model.compile(optimizer=opt(lr=self.learning_rate), loss=loss, metrics=[keras.metrics.MSE])
    model.build(input_shape = self.input_size)
    return model
    
  def get_TD_error(self, reinforcement, old_state, new_state):
    new = tf.convert_to_tensor([new_state])
    old = tf.convert_to_tensor([old_state])

    return reinforcement + self.discount_factor*self.model(old) - self.model(new)
  
  def update(self, reinforcement, old_state, new_state): 
    new = tf.convert_to_tensor([new_state])
    old = tf.convert_to_tensor([old_state])
    target = reinforcement + self.discount_factor * self.model(new) 
    self.nn.fit(old, target, verbosity=0)
