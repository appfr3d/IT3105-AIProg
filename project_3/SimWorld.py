from enum import Enum

class ShapeType(Enum):
  """
  Enum class for defining board shapes
  """
  TRIANGLE = 1
  DIAMOND = 2

class DirectionType(Enum):
  """
  Enum class for defining directions, used for defining directions of connections in board connection matrixes
  """
  UP_RIGHT = 1 # Start at 1 to not count as False
  RIGHT = 2
  DOWN_RIGHT = 3
  DOWN_LEFT = 4
  LEFT = 5
  UP_LEFT = 6
  

class SimWorldBase:
  """
  A base class for simworlds, defining the actions that reinforcement agents or other agents can expect a simworld to support. 
  An informal interface to simworlds.
  """
  def __init__(self):
    pass
    
  def display(self):
    pass
  
  def get_win(self):
    pass
  
  def get_game_over(self):
    pass
  
  def get_actions(self):
    pass

  def make_move(self, move):
    pass

  def get_log_metric(self):
    pass


class SimWorldPlayer:
  def reset_state(self):
    pass

  def get_state(self):
    pass

  def get_actions(self):
    pass

  def do_action(self, action):
    pass

  def get_game_over(self):
    pass
  
  def get_reward(self):
    pass

  def get_log_metric(self):
    pass

  def force_display_frame(self):
    pass