import os
from SimWorld import ShapeType
from Critic import CriticType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, "configs")

class ConfigReader():
  def __init__(self):
    """
    Sets values to all of the config variables to make sure all the values are initialized in case something is missing
    from the config.txt file. 
    Calls self.read_config to read the values in the config.txt file and update the config variables.
    Does a check to enshure that the empty cells are legal.
    """
    self.board_type = ShapeType.DIAMOND
    self.size = 4
    self.empty_cells = [6,7]
    self.number_of_episodes = 10

    self.critic_type = CriticType.TABLE 
    self.critic_nn_dimentions = [15, 20, 30, 5, 1]

    self.actor_learning_rate = 0.1
    self.critic_learning_rate = 0.1

    self.actor_eligibility_decay_rate = 0.8972402459603667
    self.critic_eligibility_decay_rate = 0.8931089114734135

    self.actor_discount_factor = 0.9409684975677948
    self.critic_discount_factor = 0.602227525679401

    self.initial_epsilon = 0.30862613139410333
    
    self.frame_delay = 1000
    self.image_size = 1000

    self.move_loss = 1.0
    self.peg_loss2 = 1.0
    self.base_reward = 0.0
    self.win_reward = 1000

    self.read_config()

    # Ensures that the empty cells is inside bounds
    remove_empty_cells = []
    for i in range(len(self.empty_cells)):
      if self.empty_cells[i] >= self.size*self.size:
        print('Warning: empty_cell out of bounds.', self.empty_cells[i], '> '+ str(self.size*self.size - 1) + '. Cell ignored.')
        remove_empty_cells.append(self.empty_cells[i])
    for cell in remove_empty_cells:
      self.empty_cells.remove(cell)
      

  def read_config(self):
    """
    Reads the config.txt file and saves the values in the corresponding variables
    """
    # Ask the use which config file to use
    config_files = [f for f in os.listdir(CONFIG_DIR) if f.startswith("config") and f.endswith(".txt")]
    print('Which config file do you want to use?:')
    for i in range(len(config_files)):
      print('(' + str(i) + '): ' + config_files[i])
    file_index = input('(0-'+str(len(config_files)-1)+'): ')
    while not file_index.isdigit() or int(file_index) < 0 or int(file_index) > (len(config_files)-1):
      file_index = input('(0-'+str(len(config_files)-1)+'): ')
  
    file_name = config_files[int(file_index)]

    f = open(os.path.join(CONFIG_DIR, file_name))
    for line in f:
      if line.strip() == '' or line.startswith('#'):
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
      
      elif key == 'empty_cells':
        self.empty_cells = [int(c) for c in val.split(',')]

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
        # In millisecounds
        self.frame_delay = int(val)
      
      elif key == 'image_size':
        self.image_size = int(val)

      elif key == 'win_reward':
        self.win_reward = float(val)
      elif key == 'base_reward':
        self.base_reward = float(val)

      elif key == 'peg_loss':
        self.peg_loss = float(val)

      elif key == 'peg_loss2':
        self.peg_loss2 = float(val)

      elif key == 'move_loss':
        self.move_loss = float(val)

      elif key == 'epsilon_decay_rate':
        self.epsilon_decay_rate = float(val)
