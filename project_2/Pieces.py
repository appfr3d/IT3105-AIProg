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
    EMPTY = 0  # Last moved
    PLAYER1 = 1  # Not removed but not last moved
    PLAYER2 = 2  # Removed

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
    self.color_map = {PegState.PLAYER1: (0, 0, 255), PegState.PLAYER2: (255, 0, 0), PegState.EMPTY: (200, 200, 200)}
  

  def get_color(self):
    """
    :returns: A color based on the PegState of the piece
    """
    return self.color_map[self.state]
  
  def change_state(self, newState: PegState):
    """
    Sets the Peg state of the piece
    :param PegState: The peg state to change to
    """
    self.state = newState

  def player1_place_peg(self):
    self.change_state(PegState.PLAYER1)

  def player2_place_peg(self):
    self.change_state(PegState.PLAYER2)
