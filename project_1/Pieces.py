from enum import Enum
import typing

class Piece:
  def __init__(self, position: tuple):
    self.posistion = position # (row, col)

class PegState(Enum):
    SELECTED = 0
    UNSELECTED = 1
    REMOVED = 2
# A peg is a piece that has a position, and that can be selected, unselected, and removed
class Peg(Piece):
  def __init__(self, position: tuple, state: PegState):
    super(Peg, self).__init__(position)
    self.state = state
    self.colorMap = {PegState.SELECTED: [255, 0, 0], PegState.UNSELECTED: [0, 0, 0], PegState.REMOVED: [255, 255, 255]}
  
  

  def display(self):
      return self.colorMap[self.state]
  
  def changeState(self, newState: PegState):
    self.state = newState
