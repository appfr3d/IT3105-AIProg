from simWorld import HexBoard
from simWorld import ShapeType
from Pieces import Peg, PegState
from SimWorldDisplayer import Line, Circle, Layer

class PegSolitaireBoard(HexBoard):
  def __init__(self, initial_state_data, shape, size):
    super(PegSolitaireBoard, self).__init__(initial_state_data, shape, size)
    self.init_board_pieces()
  
  def init_board_pieces(self):
    if self.shape == ShapeType.DIAMOND:
      # If shape is diamond make every position in board be filled up by a peg
      for row_num in range(len(self.board)):
        row = self.board[row_num]
        for col_num in range(len(row)):
          row[col_num] = Peg(position=(row_num, col_num), state=PegState.UNSELECTED)
          
      # Then remove positions which should be empty
      # Where a position is an integer giving the index that a peg should be,
      # in the case of diamonds counting from top and to the right after rotating 45 degrees counter clockwise
      empty_positions = self.initial_state_data.getEmptyPositions() 
      
      for position in empty_positions: 
        row = position//self.size
        col = position%self.size
        self.board[row][col].changeState(PegState.REMOVED)

    if self.shape == ShapeType.TRIANGLE:
      for row_num in range(len(self.board)):
        row = self.board[row_num]
        for col_num in range(len(row)):
          # This check is also used in the initialization of the connection matrix, basically it just skips everything 
          # to the right of the diagonal from top left to bottom right in the matrix,
          # only change is mapping 2d index to imaginary 1d index first
          node_i = row_num*self.size + col_num
          if node_i%self.size >= self.size - (self.size - node_i//self.size - 1):
            continue  # Every position is False by default, so everything outside of triangle is still False by not being set to anything else

          row[col_num] = Peg(position=(row_num, col_num), state=PegState.UNSELECTED)
      
      # Where a position is an integer giving the index that a peg should be,
      # in the case of triangles counting from top and then each row from the left before any adjustment
      empty_positions = self.initial_state_data.getEmptyPositions() 
      
      max_position = max(empty_positions)

      # We have a position index and a matrix index because we need to skip over some empty positions in the matrix, while still counting
      # how many of the non-empty positions we've visited
      position_index = 0 # position index is an index into a hypothethical 1D list with only valid positions

      # We use this index to iterate through the matrix as if it was one dimensional by using // and %
      matrix_index = 0

      while position_index <= max_position:
        if matrix_index%self.size >= self.size - (self.size - matrix_index//self.size - 1):
          # Check if index is on wrong side of diagonal
          # If it is on wrong side go to next matrix entry, but don't iterate position index
          matrix_index += 1 

        elif position_index in empty_positions:
           # If on right side of diagonal, and at right position, remove piece at that position
          self.board[matrix_index//self.size][matrix_index%self.size].changeState(PegState.REMOVED) 
          matrix_index += 1
          position_index += 1
        
        else:
          # If position is a valid position in matrix, but not a piece we want to remove, move on to next
          matrix_index += 1 
          position_index += 1

  def display(self, image_size):
    connection_layer = self.draw_connection_matrix(image_size)
    node_layer = self.draw_node_layer(image_size)
    return [connection_layer, node_layer]

  def draw_connection_matrix(self, image_size):
    return False 

  def draw_node_layer(self, image_size):
    return False