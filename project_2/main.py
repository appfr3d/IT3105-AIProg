from ConfigReader import ConfigReader
from SimWorldDisplayer import ImageDisplay
from HexGameBoard import HexGameBoard
from ReinforcementLearner import ReinforcementLearner
from simWorld import ShapeType
import random
import os
from tqdm import tqdm
from PlayerEnum import Player
from MonteCarloTreeNodes import HexGameBridge
from ActorNN import HexBoardNNBridge
from Tournament import Tournament
import cProfile
import re
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Create config object
config = ConfigReader()

# train, train_multiple or tournament
run_type = 'tournament'

if run_type == 'train':
  # Read config file
  file_name = config.read_config()

  # Initialize bridges
  nn_bridge = HexBoardNNBridge(config)
  game_bridge = HexGameBridge(config)

  # Create save path
  t = str(round(time.time(), 4)).replace('.', '')
  save_path = os.path.join(CURRENT_DIR, 'tournament_models', t)
  
  # Initialize the RL learner
  learner = ReinforcementLearner(config, nn_bridge, game_bridge, save_path)

  # Train the learner
  learner.fit()

  # Copy config file to new folder
  with open(os.path.join(CURRENT_DIR, 'configs', file_name)) as old_config:
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    with open(os.path.join(save_path, 'config_used.txt'), 'w+') as new_config:
      for line in old_config:
        new_config.write(line)

elif run_type == 'train_multiple':
  # Get config file names
  file_names = config.get_config_files()

  # Train on every config file
  for f in file_names:
    print('Training on config file: ' + f)
    # Read config file
    config.read_config(f)

    # Initialize bridges
    nn_bridge = HexBoardNNBridge(config)
    game_bridge = HexGameBridge(config)

    # Create save path
    t = str(round(time.time(), 4)).replace('.', '')
    save_path = os.path.join(CURRENT_DIR, 'tournament_models', t)

    # Initialize the RL learner
    learner = ReinforcementLearner(config, nn_bridge, game_bridge, save_path)

    # Train the learner
    learner.fit()

    # Copy config file to new folder
    with open(os.path.join(CURRENT_DIR, 'configs', f)) as old_config:
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      with open(os.path.join(save_path, 'config_used.txt'), 'w+') as new_config:
        for line in old_config:
          new_config.write(line)


elif run_type == 'tournament':
  # Find the folder where the tournament data lies
  configs_dir = os.path.join(CURRENT_DIR, 'tournament_models')
  config_folders = [name for name in os.listdir(configs_dir) if os.path.isdir(os.path.join(configs_dir, name))]

  if len(config_folders) > 1:
    print('Which config file do you want to use?:')
    for i in range(len(config_folders)):
      print('(' + str(i) + '): ' + config_folders[i])
    folder_index = input('(0-'+str(len(config_folders)-1)+'): ')
    while not folder_index.isdigit() or int(folder_index) < 0 or int(folder_index) > (len(config_folders)-1):
      folder_index = input('(0-'+str(len(config_folders)-1)+'): ')

    folder_name = config_folders[int(folder_index)]
  else:
    folder_name = config_folders[0]

  choosen_tournament_folder = os.path.join(configs_dir, folder_name)

  # Read config file
  config.read_config(os.path.join(choosen_tournament_folder, 'config_used.txt'))

  game_bridge = HexGameBridge(config)
  nn_bridge = HexBoardNNBridge(config)
  
  tourney = Tournament(config, choosen_tournament_folder, game_bridge, nn_bridge)
  tourney.run_tourney()


