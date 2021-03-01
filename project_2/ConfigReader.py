import os
from simWorld import ShapeType
from ActivationFunctionType import ActivationFunctionType
from OptimizerType import OptimizerType

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
    self.M = 5

    self.exploration_constant = 1

    self.number_of_episodes = 10
    self.rollouts_per_move = 10

    self.actor_learning_rate = 0.1
    self.neurons_per_layer = [self.size*self.size]
    self.activation_functions_per_layer = ["RELU"]

    self.tournament_participants = 10
    self.tournament_games = 10

    self.frame_delay = 1000
    self.image_size = 1000

    self.read_config()

      

  def read_config(self):
    """
    Reads the config.txt file and saves the values in the corresponding variables
    """
    # Ask the use which config file to use
    config_files = [f for f in os.listdir(CONFIG_DIR) if f.startswith("config") and f.endswith(".txt")]
    if len(config_files) > 1:
      print('Which config file do you want to use?:')
      for i in range(len(config_files)):
        print('(' + str(i) + '): ' + config_files[i])
      file_index = input('(0-'+str(len(config_files)-1)+'): ')
      while not file_index.isdigit() or int(file_index) < 0 or int(file_index) > (len(config_files)-1):
        file_index = input('(0-'+str(len(config_files)-1)+'): ')

      file_name = config_files[int(file_index)]
    else:
      file_name = config_files[0]

    f = open(os.path.join(CONFIG_DIR, file_name))
    for line in f:
      if line.strip() == '' or line.startswith('#'):
        continue

      parts = line.strip().split(':')
      
      key = parts[0].strip()
      val = parts[1].strip()

      
      if key == 'size':
        self.size = int(val)
      
      elif key == 'M':
        self.M = int(val)
      
      elif key == 'exploration_constant':
        self.exploration_constant = float(val)

      elif key == 'tournament_participants':
        self.tournament_participants = int(val)

      elif key == 'tournament_games':
        self.tournament_games = int(val)

      elif key == 'number_of_episodes':
        self.number_of_episodes = int(val)

      elif key == 'rollouts_per_move':
        self.number_of_episodes = int(val)

      elif key == 'neurons_per_layer':
        self.neurons_per_layer = [int(d) for d in val.split(',')]

      elif key == 'actor_learning_rate':
        self.actor_learning_rate = float(val)

      elif key == 'frame_delay':
        # In millisecounds
        self.frame_delay = int(val)
      
      elif key == 'image_size':
        self.image_size = int(val)
