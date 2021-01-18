from ConfigReader import ConfigReader
from SimWorldDisplayer import ImageDisplay
from pegSolitaireBoard import PegSolitaireBoard
from simWorld import ShapeType
conf = ConfigReader()

board = PegSolitaireBoard(conf.empty_cells, conf.board_type, conf.size)

print(board.get_all_moves())

img_display = ImageDisplay(board)
img_display.display()