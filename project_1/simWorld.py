from enum import Enum

class ShapeType(Enum):
  TRIANGLE = 1
  DIAMOND = 2

class SimWorldBase:
  def __init__(self, initial_state_data):
    self.initial_state_data = initial_state_data  # Contains which positions are empty
    
  def display(self):
    pass

class HexBoard(SimWorldBase):
  def __init__(self, initial_state_data, shape, size):
    super(HexBoard, self).__init__(initial_state_data)
    self.shape = shape
    self.size = size
    self.init_board(size)
  
  def init_board(self, size):
    self.board =  [[False for i in range(size)] for j in range(size)]  # One entry for every node, if diamond all will be filled with pieces, if triange half of matrix including diagonal from 
    # top left to bottom right will be filled

    # One entry for every node pair (i, j), where cM(i, j) = True if there is a connection from i to j. (i, i) 
    self.connection_matrix = [[False for i in range(size*size)] for j in range(size*size)]
    # does not have a connection
    if self.shape == ShapeType.DIAMOND:
      for node_i in range(size*size):
        top_boundry = node_i < size  # Checks if node is on top of board
        left_boundry = node_i % size == 0  # Check if node is in leftmost column in board
        right_boundry = (node_i + 1) % size == 0  # Check if node is in rightmost column in board
        bottom_boundry = node_i > size*size-1-size  # Check if node is in bottommost coulmn in board
        
        # See docs/Diamond_Connection_Matrix.png for visualization
        if not top_boundry:
          self.connection_matrix[node_i][node_i-size] = True
        if not top_boundry and not right_boundry:
          self.connection_matrix[node_i][node_i-size+1] = True
        if not right_boundry:
          self.connection_matrix[node_i][node_i+1] = True
        if not bottom_boundry:
          self.connection_matrix[node_i][node_i+size] = True
        if not bottom_boundry and not left_boundry:
          self.connection_matrix[node_i][node_i+size-1] = True
        if not left_boundry:
          self.connection_matrix[node_i][node_i-1] = True
            
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
          self.connection_matrix[node_i][node_i-size] = True
        if not right_boundry and not diagonal_boundry:
          self.connection_matrix[node_i][node_i+1] = True
        if not right_boundry and not bottom_boundry:
          self.connection_matrix[node_i][node_i+size+1] = True
        if not bottom_boundry:
          self.connection_matrix[node_i][node_i+size] = True
        if not left_boundry:
          self.connection_matrix[node_i][node_i-1] = True
        if not left_boundry and not top_boundry:
          self.connection_matrix[node_i][node_i-size-1] = True
    
  def display(self):
    pass
