from ConfigReader import ConfigReader
from SimWorldDisplayer import ImageDisplay
from PegSolitaireBoard import PegSolitaireBoard
from SimWorld import ShapeType
from ReinforcementLearner import ReinforcementLearner
from PegSolitairePlayer import PegSolitairePlayer

config = ConfigReader()

player = PegSolitairePlayer(config.empty_cells, config.board_type, config.size, config.image_size, config.frame_delay)
learner = ReinforcementLearner(player, config)

learner.fit()
print('Number of correct runs:', sum([1 for ele in learner.peg_log if ele == 1]))
learner.display_game()




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