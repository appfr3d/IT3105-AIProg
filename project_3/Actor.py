from enum import Enum
from tensorflow import keras
import tensorflow as tf
import random
import ConfigReader
import SplitGD
import numpy as np


class Actor():
  """
  abstract actor class
  """
  def eligibility_decay(self, state):
    pass
  
  def value_update(self, state, TD_error):
    pass

  def set_eligibility(self, state, value):
    pass
  
  def get_TD_error(self, old_state, new_state):
    pass


class NNActor(Actor):
  def __init__(self, config: ConfigReader):
    self.epsilon = config.initial_epsilon
    self.learning_rate = config.critic_learning_rate
    self.eligibility_decay_rate = config.critic_eligibility_decay_rate
    self.discount_factor = config.critic_discount_factor
    self.nn_dimensions = config.critic_nn_dimentions
    self.config = config

    self.model = self.generate_fully_connected()
    self.nn = SplitGD.ReinforcementGD(self.model, self.discount_factor, self.eligibility_decay_rate)

    self.model.summary()

  def reset_eligibility(self):
    self.nn.eligibility_gradients = None
  
  def generate_fully_connected(self):
    """
    :returns a fully connected Sequential keras model with layers as defined in the config file
    """
    opt = keras.optimizers.SGD  # Stocastic Gradient Decent
    loss = keras.losses.MSE     # Mean Square Error
    model = keras.models.Sequential()

    # Add fully connected layers to the model
    for num in range(len(self.nn_dimensions)):
      dim = self.nn_dimensions[num]
      if num == 0:
        model.add(keras.layers.Input(dim))
      else:
        model.add(keras.layers.Dense(dim, activation=self.config.activation_function))

    # Mean Square Error (MSE) metric works well with calculating TD-error by using target as in update function
    model.compile(optimizer=opt(lr=self.learning_rate), loss=loss, metrics=[keras.metrics.MSE])
    return model
    
  def get_TD_error(self, reinforcement, old_state, new_state):
    new = tf.convert_to_tensor([new_state])
    old = tf.convert_to_tensor([old_state])

    return reinforcement + self.discount_factor*self.model(old) - self.model(new)
  
  def update(self, reinforcement, old_state, new_state):
    new = tf.convert_to_tensor([new_state])
    old = tf.convert_to_tensor([old_state])
    # Here target used with MSE corresponds to TD error on derivation (based on handout material)
    target = reinforcement + self.discount_factor * self.model(new)
    self.nn.fit(old, target, verbosity=0)
  
  def select_action(self, nn_input):
    vals = []
    for action in nn_input:
      vals.append(np.array(self.model(action))[0, 0])
    if random.random() <= self.epsilon:
      r_indx = random.randint(0, len(vals)-1)
    else:
      #print(vals)
      vals2 = np.asarray(vals)
      # to avoid always taking the same action when all vals are 0 (happens rapidly)
      # max(vals) to avoid an issue with very small floating point numbers that showed up...
      if len(vals2[vals2 == max(vals)]) == 3 or max(vals) < 0.000000001:
        r_indx = vals.index(max(vals))
      elif len(vals2[vals2 == max(vals)]) == 1:
        r_indx = random.randint(0, len(vals) - 1)
      else:
        if random.random() >= 0.5:
          r_indx = vals.index(max(vals))
        else:
          vals.reverse()
          r_indx = vals.index(max(vals))
    return r_indx, nn_input[r_indx]

