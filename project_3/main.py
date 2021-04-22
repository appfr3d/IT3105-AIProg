from ConfigReader import ConfigReader
from SimWorldDisplayer import ImageDisplay
from SimWorld import ShapeType
from ReinforcementLearner import ReinforcementLearner
from CarPlayer import CarPlayer
import random
import os
import numpy as np
from tqdm import tqdm
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
  pglg = np.asarray(learner.peg_log)
  save_file = open('config_random.txt', 'a')
  save_file.write(str(len(list(pglg[pglg == 1000]))) + ", ")
  save_file.write(str(config.critic_learning_rate) + ", ")
  save_file.write(str(config.critic_discount_factor) + ", ")
  save_file.write(str(config.initial_epsilon) + ", ")
  save_file.write(str(config.epsilon_decay_rate) + ", ")
  save_file.write(str(config.win_reward) + ", ")
  save_file.write(str(config.base_reward) + ", ")
  save_file.write(str(config.tiles) + ", ")
  save_file.write(str(config.tiles_per_tile) + ", ")
  save_file.write(str(config.critic_nn_dimentions) + ", ")
  save_file.write(str(config.y) + ", ")
  save_file.write(str(config.velocity) + ", ")
  save_file.write(str(config.tiling_offset) + ", ")
  save_file.write("\n")
  save_file.close()



  #learner.display_game()


# FOR TESTING:
#res = []
#for num in tqdm(range(0, 5), desc="Progress"):
#  player = PegSolitairePlayer(config.empty_cells, config.board_type, config.size, config.image_size, config.frame_delay, config.win, config.base_loss, config.peg_loss, config.peg_loss2, config.move_loss)
#  learner = ReinforcementLearner(player, config)
#
#  learner.fit()
#  res.append(sum([1 for ele in learner.peg_log if ele == 1]))
#
#print(res)
#avg = sum(res)/len(res)
#print(avg)
#res2 = [abs(x - avg)**2 for x in res]
#std = sum(res2)/len(res)
#print(std)
#conf = open("works well for triangle nn (not tested diamond, but that one worked well already).txt", "r")
#text = "".join(conf.readlines())
#
#conf.close()
#with open("log.txt", "a") as file:
#  file.write(str(res) + " | " + str(avg) + " | " + str(std) + " \n " + text + "\n" + "--------------")


