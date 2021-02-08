from ConfigReader import ConfigReader
from SimWorldDisplayer import ImageDisplay
from PegSolitaireBoard import PegSolitaireBoard
from SimWorld import ShapeType
from ReinforcementLearner import ReinforcementLearner
from PegSolitairePlayer import PegSolitairePlayer
import random
from tqdm import tqdm


config = ConfigReader()
player = PegSolitairePlayer(config.empty_cells, config.board_type, config.size, config.image_size, config.frame_delay, config.win, config.base_loss, config.peg_loss, config.peg_loss2, config.move_loss)
learner = ReinforcementLearner(player, config)

learner.fit()
print('Number of correct runs:', sum([1 for ele in learner.peg_log if ele == 1]))
learner.display_game()

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


'''
# board = PegSolitaireBoard(config.empty_cells, config.board_type, config.size)
img_display = ImageDisplay(board)
img_display.display()

moves = board.get_all_moves()


print(moves)

board.do_action(moves[0])

img_display.display()

# img_display.display()
'''