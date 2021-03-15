from ConfigReader import ConfigReader
from SimWorldDisplayer import ImageDisplay
from HexGameBoard import HexGameBoard
from ReinforcementLearner import ReinforcementLearner
from simWorld import ShapeType
import random
from tqdm import tqdm
from PlayerEnum import Player
from MonteCarloTreeNodes import HexGameBridge
from ActorNN import HexBoardNNBridge

# Read config file
config = ConfigReader()

# Initialize bridges
nn_bridge = HexBoardNNBridge(config)
game_bridge = HexGameBridge(config)

# Initialize the RL learner
learner = ReinforcementLearner(config, nn_bridge, game_bridge)

# Train the learner
learner.fit()
