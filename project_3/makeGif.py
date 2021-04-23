import os
from ConfigReader import ConfigReader
from CarPlayer import CarPlayer
from ReinforcementLearner import ReinforcementLearner

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_choosen_model_folder():
  # Find the folder where the tournament data lies
  configs_dir = os.path.join(CURRENT_DIR, 'models')
  config_folders = [name for name in os.listdir(configs_dir) if os.path.isdir(os.path.join(configs_dir, name))]

  if len(config_folders) > 1:
    print('Which model do you want to use?:')
    for i in range(len(config_folders)):
      print('(' + str(i) + '): ' + config_folders[i])
    folder_index = input('(0-' + str(len(config_folders) - 1) + '): ')
    while not folder_index.isdigit() or int(folder_index) < 0 or int(folder_index) > (len(config_folders) - 1):
      folder_index = input('(0-' + str(len(config_folders) - 1) + '): ')

    folder_name = config_folders[int(folder_index)]
  else:
    folder_name = config_folders[0]

  return os.path.join(configs_dir, folder_name)


# Read model path
model_folder = get_choosen_model_folder()
config_file = os.path.join(model_folder, 'config_used.txt')

config = ConfigReader(config_file)

player = CarPlayer(config)
learner = ReinforcementLearner(player, config)
# learner.fit()
learner.display_game()

player.sim_world_displayer.save_gif(model_folder)