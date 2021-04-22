import os

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
    self.number_of_episodes = 10

    self.critic_nn_dimentions = [15, 20, 30, 5, 1]

    self.critic_learning_rate = 0.1

    self.critic_eligibility_decay_rate = 0.8931089114734135

    self.critic_discount_factor = 0.602227525679401

    self.initial_epsilon = 0.30862613139410333
    self.epsilon_decay_rate = 0.99

    self.frame_delay = 1000
    self.image_size = 1000

    self.base_reward = 0.0
    self.win_reward = 1000

    self.tiles = 5
    self.tiles_per_tile = 5
    self.tiling_offset = 0.1

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
            
      if key == 'number_of_episodes':
        self.number_of_episodes = int(val)

      elif key == 'critic_nn_dimentions':
        self.critic_nn_dimentions = [int(d) for d in val.split(',')]
            
      elif key == 'critic_learning_rate':
        self.critic_learning_rate = float(val)

      elif key == 'critic_eligibility_decay_rate':
        self.critic_eligibility_decay_rate = float(val)

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

      elif key == 'epsilon_decay_rate':
        self.epsilon_decay_rate = float(val)

      elif key == 'tiles':
        self.tiles = int(val)
      
      elif key == 'tiles_per_tile':
        self.tiles_per_tile = int(val)
      
      elif key == 'tiling_offset':
        self.tiling_offset = float(val)
      
