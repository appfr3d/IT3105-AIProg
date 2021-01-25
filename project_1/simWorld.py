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

class HexBoard(SimWorldBase):
  """
  A semi-abstract SimWorld type, used as a super-class of concrete implementations of Hexagonal game boards. 
  """
  def __init__(self, empty_positions, shape, size):
    """
    :param empty_positions: A list of board positions that should be empty, see /docs/Diamond_Connection_Matrix and /docs/Triangle_Connection_Matrix 
    for counting scheme
    :param shape: A ShapeType defining the shape of the hexagonal game board
    :param size: The width of the diagonal of the game board matrix
    """
    self.empty_positions = empty_positions  # Contains which positions are empty
    self.shape = shape
    self.size = size
    self.init_board(size)
  
  def init_board(self, size):
    """
    Initializes self.board with pegs int the correct positions, based on self.shape
    :param size: the width and height of self.board
    """
    # One entry for every node, if diamond all will be filled with pieces, if triange half of matrix including 
    # diagonal from top left to bottom right will be filled
    self.board =  [[False for i in range(size)] for j in range(size)]  

    # One entry for every node pair (i, j), where cM(i, j) = direction enum if there is a connection from i to j. 
    # (i, i) does not have a connection
    self.connection_matrix = [[False for i in range(size*size)] for j in range(size*size)]
    if self.shape == ShapeType.DIAMOND:
      for node_i in range(size*size):
        top_boundry = node_i < size                 # Check if node is on top of board
        left_boundry = node_i % size == 0           # Check if node is in leftmost column in board
        right_boundry = (node_i + 1) % size == 0    # Check if node is in rightmost column in board
        bottom_boundry = node_i > size*size-1-size  # Check if node is in bottommost coulmn in board
        
        # See docs/Diamond_Connection_Matrix.png for visualization
        if not top_boundry:
          self.connection_matrix[node_i][node_i-size] = DirectionType.UP_RIGHT
        if not top_boundry and not right_boundry:
          self.connection_matrix[node_i][node_i-size+1] = DirectionType.RIGHT
        if not right_boundry:
          self.connection_matrix[node_i][node_i+1] = DirectionType.DOWN_RIGHT
        if not bottom_boundry:
          self.connection_matrix[node_i][node_i+size] = DirectionType.DOWN_LEFT
        if not bottom_boundry and not left_boundry:
          self.connection_matrix[node_i][node_i+size-1] = DirectionType.LEFT
        if not left_boundry:
          self.connection_matrix[node_i][node_i-1] = DirectionType.UP_LEFT
            
    elif self.shape == ShapeType.TRIANGLE:
      for node_i in range(size*size):
        # check if node_i is in the empty triangle. 
        # No proof for this but some sketching suggested the formula, and the formula worked with empirical testing
        triangle_check = node_i%size >= size - (size - node_i//size - 1)  
        # for many different sizes
        # == gives on diagonal to the right of main diagonal through matrix, greater gives the numbers on the rest of the row
        # basic intuition: size-node_i//size-1 gives how many of the nodes on a row in the board matrix are empty, 
        # and the rest checks if the node_i is in such an area
        if triangle_check:  # If it is in the empty side there should be no connections so skip ahead
          continue

        top_boundry = node_i < size  # Checks if node is on top of board
        left_boundry = node_i % size == 0  # Check if node is in leftmost column in board
        right_boundry = (node_i + 1) % size == 0  # Check if node is in rightmost column in board
        bottom_boundry = node_i > size*size-1-size  # Check if node is in bottommost coulmn in board
        diagonal_boundry = node_i%(size+1) == 0 # Check if node is on diagonal in board

        # See docs/Triangle_Connection_Matrix.png for visualization
        if not top_boundry and not diagonal_boundry:
          self.connection_matrix[node_i][node_i-size] = DirectionType.UP_RIGHT
        if not right_boundry and not diagonal_boundry:
          self.connection_matrix[node_i][node_i+1] = DirectionType.RIGHT
        if not right_boundry and not bottom_boundry:
          self.connection_matrix[node_i][node_i+size+1] = DirectionType.DOWN_RIGHT
        if not bottom_boundry:
          self.connection_matrix[node_i][node_i+size] = DirectionType.DOWN_LEFT
        if not left_boundry:
          self.connection_matrix[node_i][node_i-1] = DirectionType.LEFT
        if not left_boundry and not top_boundry:
          self.connection_matrix[node_i][node_i-size-1] = DirectionType.UP_LEFT
    
  def display(self):
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