from enum import Enum
import typing

class Piece:
  """
  Generic class for all pieces
  """
  def __init__(self, position: tuple):
    """
    :param position: The position of the piece
    """
    self.posistion = position # (row, col)

class PegState(Enum):
    SELECTED = 0  # Last moved
    UNSELECTED = 1  # Not removed but not last moved
    REMOVED = 2  # Removed
# A peg is a piece that has a position, and that can be selected, unselected, and removed
class Peg(Piece):
  """
  A peg is a type of board piece, which can be selected, not selected, or removed.
  """
  def __init__(self, position: tuple, state: PegState):
    """
    :param position: The position of the piece
    :param state: The PegState of the piece
    """
    super(Peg, self).__init__(position)
    self.state = state
    self.color_map = {PegState.SELECTED: (255, 0, 0), PegState.UNSELECTED: (0, 0, 255), PegState.REMOVED: (200, 200, 200)}
  

  def get_color(self):
    """
    :returns: A color based on the PegState of the piece
    """
    return self.color_map[self.state]
  
  def changeState(self, newState: PegState):
    """
    Sets the Peg state of the piece
    :param PegState: The peg state to change to
    """
    self.state = newState
