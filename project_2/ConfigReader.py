import os
from simWorld import ShapeType
from ActivationFunctionType import ActivationFunctionType
from OptimizerType import OptimizerType
import tensorflow as tf
from tensorflow import keras

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, "configs")
MAIN_CONFIG_DIR = os.path.join(CURRENT_DIR, "main_configs")



class ConfigReader():
  def __init__(self):
    """
    Sets values to all of the config variables to make sure all the values are initialized in case something is missing
    from the config.txt file. 
    Calls self.read_config to read the values in the config.txt file and update the config variables.
    Does a check to enshure that the empty cells are legal.
    """
    self.run_type = 'train'
    self.board_type = ShapeType.DIAMOND
    self.size = 4
    self.model_count = 5
    self.games_per_series = 25

    self.exploration_constant = 1

    self.initial_epsilon = 0.75
    self.epsilon_decay_rate = 0.99

    self.number_of_episodes = 10
    self.rollouts_per_move = 10

    self.actor_learning_rate = 0.1
    self.neurons_per_layer = [self.size*self.size]

    self.frame_delay = 1000
    self.image_size = 1000
    self.optimizer = keras.optimizers.SGD
    self.activation_func = 'relu'

    self.tournament_action_mode = 'greedy'

    # Read teh main config file
    self.read_main_config()

  def get_main_config_files(self):
    """
    Returns the config file names
    """
    return [f for f in os.listdir(CONFIG_DIR) if f.startswith("main_config") and f.endswith(".txt")]

  def read_main_config(self):
    """
    Reads a main_config_x.txt file and saves the values in the corresponding variables
    """
    # Ask the use which config file to use
    config_files = self.get_main_config_files()
    if len(config_files) > 1:
      print('Which main config file do you want to use?:')
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

      if key == 'run_type':
        self.run_type = val

      elif key == 'games_per_series':
        self.games_per_series = int(val)
      
      elif key == 'tournament_action_mode':
        self.tournament_action_mode = val
      
      elif key == 'frame_delay':
        # In millisecounds
        self.frame_delay = int(val)
      
      elif key == 'image_size':
        self.image_size = int(val)


  def get_config_files(self):
    """
    Returns the config file names
    """
    return [f for f in os.listdir(CONFIG_DIR) if f.startswith("config") and f.endswith(".txt")]

  def read_config(self, file_path=None):
    """
    Reads a config_x.txt file and saves the values in the corresponding variables
    """
    if file_path == None:
      # Ask the use which config file to use
      config_files = self.get_config_files()
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
    else:
      # Else use input file path
      file_name = file_path

    f = open(os.path.join(CONFIG_DIR, file_name))
    for line in f:
      if line.strip() == '' or line.startswith('#'):
        continue

      parts = line.strip().split(':')
      
      key = parts[0].strip()
      val = parts[1].strip()

      if key == 'size':
        self.size = int(val)
      
      elif key == 'model_count':
        self.model_count = int(val)
      
      elif key == 'exploration_constant':
        self.exploration_constant = float(val)
      
      elif key == 'initial_epsilon':
        self.initial_epsilon = float(val)
      
      elif key == 'epsilon_decay_rate':
        self.epsilon_decay_rate = float(val)

      elif key == 'number_of_episodes':
        self.number_of_episodes = int(val)

      elif key == 'rollouts_per_move':
        self.rollouts_per_move = int(val)

      elif key == 'neurons_per_layer':
        self.neurons_per_layer = [int(d) for d in val.split(',')]

      elif key == 'actor_learning_rate':
        self.actor_learning_rate = float(val)
      
      elif key == 'optimizer':
        if val == 'ADAGRAD':
          self.optimizer = keras.optimizers.Adagrad
        elif val == 'SGD':
          self.optimizer = keras.optimizers.SGD
        elif val == 'RMSPROP':
          self.optimizer = keras.optimizers.RMSprop
        elif val == 'ADAM':
          self.optimizer = keras.optimizers.Adam
      
      elif key == 'activation_func':
        if val == 'LINEAR':
          self.activation_func = 'linear'
        elif val == 'RELU':
          self.activation_func = 'relu'
        elif val == 'SIGMOID':
          self.activation_func = 'sigmoid'
        elif val == 'TANH':
          self.activation_func = 'tanh'

      elif key == 'rbuf_size':
        self.rbuf_size = int(val)
      elif key == 'epochs_per_rbuf':
        self.epochs_per_rbuf = int(val)

    if file_path == None:
      return file_name