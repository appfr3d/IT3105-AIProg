from ConfigReader import ConfigReader
from SimWorldDisplayer import ImageDisplay
from HexGameBoard import HexGameBoard
from simWorld import ShapeType
import random
from tqdm import tqdm
from PlayerEnum import Player


config = ConfigReader()
# player = PegSolitairePlayer(config.empty_cells, config.board_type, config.size, config.image_size, config.frame_delay, config.win_reward, config.base_reward, config.peg_loss, config.peg_loss2, config.move_loss)
board = HexGameBoard(config.board_type, config.size)


img = ImageDisplay(board)
img.display(1000)
board.do_action(((0, 0), Player.PLAYER2))
img.display(1000)
board.do_action(((1, 0), Player.PLAYER2))
img.display(1000)
board.do_action(((2, 0), Player.PLAYER2))
img.display(1000)
board.do_action(((3, 0), Player.PLAYER2))
img.display(1000)


print(board.get_win())

# learner = ReinforcementLearner(player, config)


