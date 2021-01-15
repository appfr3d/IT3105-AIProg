from simWorld import ShapeType

from enum import Enum

# TODO: Flytt denne til critic klassen når den er laget
class CriticType(Enum):
  TABLE = 1
  NN = 2

class ConfigReader():
  def __init__(self):
    self.board_type = ShapeType.DIAMOND
    self.size = 4
    self.open_cells = [6,7]
    self.number_of_episodes = 10

    self.critic_type = CriticType.TABLE 
    self.critic_nn_dimentions = [15, 20, 30, 5, 1]

    self.actor_learning_rate = 0.1
    self.critic_learning_rate = 0.1

    self.actor_eligibility_decay_rate = 0.5
    self.critic_eligibility_decay_rate = 0.5

    self.actor_discount_factor = 0.5
    self.critic_discount_factor = 0.5

    self.initial_epsilon = 0.9
    self.frame_delay = 0.5

    self.read_config()

    # print(self.board_type)
    # print(self.size)
    # print(self.open_cells)
    # print(self.number_of_episodes)
    # print(self.critic_type)
    # print(self.critic_nn_dimentions)
    # print(self.actor_learning_rate)
    # print(self.critic_learning_rate)
    # print(self.actor_eligibility_decay_rate)
    # print(self.critic_eligibility_decay_rate)
    # print(self.actor_discount_factor)
    # print(self.critic_discount_factor)
    # print(self.initial_epsilon)
    # print(self.frame_delay)

  def read_config(self):
    # Read config file
    f = open('config.txt')
    for line in f:
      if line.strip() == '':
        continue

      parts = line.strip().split(':')
      
      key = parts[0].strip()
      val = parts[1].strip()

      if key == 'board_type':
        if val == 'diamond':
          self.board_type = ShapeType.DIAMOND
        elif val == 'triangle':
          self.board_type = ShapeType.TRIANGLE
      
      elif key == 'size':
        self.size = int(val)
      
      elif key == 'open_cells':
        self.open_cells = [int(c) for c in val.split(',')]

      elif key == 'number_of_episodes':
        self.number_of_episodes = int(val)

      elif key == 'critic_type':
        if val == 'table':
          self.critic_type = CriticType.TABLE
        elif val == 'nn':
          self.critic_type = CriticType.NN

      elif key == 'critic_nn_dimentions':
        self.critic_nn_dimentions = [int(d) for d in val.split(',')]
      
      elif key == 'actor_learning_rate':
        self.actor_learning_rate = float(val)
      
      elif key == 'critic_learning_rate':
        self.critic_learning_rate = float(val)

      elif key == 'actor_eligibility_decay_rate':
        self.actor_eligibility_decay_rate = float(val)

      elif key == 'critic_eligibility_decay_rate':
        self.critic_eligibility_decay_rate = float(val)

      elif key == 'actor_discount_factor':
        self.actor_discount_factor = float(val)

      elif key == 'critic_discount_factor':
        self.critic_discount_factor = float(val)

      elif key == 'initial_epsilon':
        self.initial_epsilon = float(val)

      elif key == 'frame_delay':
        self.frame_delay = float(val)
