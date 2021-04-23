from ConfigReader import ConfigReader
from SimWorldDisplayer import ImageDisplay
from SimWorld import ShapeType
from ReinforcementLearner import ReinforcementLearner
from CarPlayer import CarPlayer
import random
import os
import numpy as np
from tqdm import tqdm

# Toggle for random testing or not
is_random_testing = False

if not is_random_testing:
  config = ConfigReader()
  player = CarPlayer(config)
  learner = ReinforcementLearner(player, config)
  learner.fit()
  learner.display_game()
else:
  while True:
    config = ConfigReader()
    config.critic_learning_rate = random.choice([0.1, 0.01 ,0.001, 0.0001, 0.00001, 0.000001])
    config.critic_discount_factor = random.random()
    config.initial_epsilon = random.random()
    config.epsilon_decay_rate = random.random()
    config.win_reward = random.random()*100
    #config.base_reward = random.choice([-0.001, 0, -0.0000001, 0.5, -1, -10, -0.1])
    config.tiles = random.randint(5, 100)
    config.tiles_per_tile = random.randint(2, 10)
    config.tiling_offset = random.choice([0.0001, 0.00001])
    config.critic_nn_dimentions = [config.tiles*config.tiles_per_tile*config.tiles_per_tile + 3, 1]
    config.critic_eligibility_decay_rate = random.random()
    config.y = random.choice([1, 0.1, 0.01, 0.001, 0.0001])
    config.velocity = random.choice([1, 0.1, 0.01, 0.001, 0.0001])
    player = CarPlayer(config)
    learner = ReinforcementLearner(player, config)

    learner.fit()



