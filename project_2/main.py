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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Read config file
config = ConfigReader()
model_save_path = os.path.join(CURRENT_DIR + '/tournament_models/' + str(config.size) + '/')

if config.run_type == 'train':
  # Initialize bridges
  nn_bridge = HexBoardNNBridge(config)
  game_bridge = HexGameBridge(config)

  # Initialize the RL learner
  learner = ReinforcementLearner(config, nn_bridge, game_bridge)

  # Train the learner
  learner.fit()

elif config.run_type == 'tournament':
  game_bridge = HexGameBridge(config)
  nn_bridge = HexBoardNNBridge(config)
  
  tourney = Tournament(config, model_save_path, game_bridge, nn_bridge)
  tourney.run_tourney()


