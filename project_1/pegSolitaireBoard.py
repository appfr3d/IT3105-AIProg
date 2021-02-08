from SimWorld import HexBoard
from SimWorld import ShapeType
from Pieces import Peg, PegState
from SimWorldDisplayer import Line, Circle, Layer
from IllegalArgumentException import * 

class PegSolitaireBoard(HexBoard):
  """
  The peg solitaire board is a subclass of the HexBoard, and is the main instance of the SimWorld superclass. It can 
  therefore be displayed by a ImageDisplay, and it models a mini-world that a reinforcement agent may act in by 
  recieving a list of possible actions, the state of the world, 
  by making actions in the world, and by recieveing game over or victory feedback.  
  """
  def __init__(self, empty_positions, shape, size):
    """
    :param empty_position: A list of positions, counted as shown in /docs/Diamond_Connection_Matrix.png and 
    /docs/Triangle_Connection_Matrix.png
    :param shape: A ShapeType
    :param size: How wide the node matrix should be across it's diagonal
    """
    super(PegSolitaireBoard, self).__init__(empty_positions, shape, size)
    self.init_board_pieces()
    self.last_selected_piece = None
  
  def init_board_pieces(self):
    """
    Initialize the board 
    """
    if self.shape == ShapeType.DIAMOND:
      # If shape is diamond make every position in board be filled up by a peg
      for row_num in range(len(self.board)):
        row = self.board[row_num]
        for col_num in range(len(row)):
          row[col_num] = Peg(position=(row_num, col_num), state=PegState.UNSELECTED)
          
      # Then remove positions which should be empty
      # Where a position is an integer giving the index that a peg should be,
      # in the case of diamonds counting from top and to the right after rotating 45 degrees counter clockwise
      
      for position in self.empty_positions: 
        row = position//self.size
        col = position%self.size
        self.board[row][col].change_state(PegState.REMOVED)

    if self.shape == ShapeType.TRIANGLE:
      for row_num in range(len(self.board)):
        row = self.board[row_num]
        for col_num in range(len(row)):
          # This check is also used in the initialization of the connection matrix, basically it just skips everything 
          # to the right of the diagonal from top left to bottom right in the matrix,
          # only change is mapping 2d index to imaginary 1d index first
          node_i = row_num*self.size + col_num
          if node_i%self.size >= self.size - (self.size - node_i//self.size - 1):
            # Every position is False by default, so everything outside of triangle is still False by not 
            # being set to anything else
            continue  

          row[col_num] = Peg(position=(row_num, col_num), state=PegState.UNSELECTED)
      
      # Where a position is an integer giving the index that a peg should be,
      # in the case of triangles counting from top and then each row from the left before any adjustment
      if len(self.empty_positions) > 0:
        max_position = max(self.empty_positions)
      else: 
        max_position = 0  

      # We have a position index and a matrix index because we need to skip over some empty positions in the matrix, 
      # while still counting how many of the non-empty positions we've visited
      # position index is an index into a hypothethical 1D list with only valid positions
      position_index = 0

      # We use this index to iterate through the matrix as if it was one dimensional by using // and %
      matrix_index = 0

      while position_index <= max_position:
        if matrix_index%self.size >= self.size - (self.size - matrix_index//self.size - 1):
          # Check if index is on wrong side of diagonal
          # If it is on wrong side go to next matrix entry, but don't iterate position index
          matrix_index += 1 

        elif position_index in self.empty_positions:
           # If on right side of diagonal, and at right position, remove piece at that position
          self.board[matrix_index//self.size][matrix_index%self.size].change_state(PegState.REMOVED) 
          matrix_index += 1
          position_index += 1
        
        else:
          # If position is a valid position in matrix, but not a piece we want to remove, move on to next
          matrix_index += 1 
          position_index += 1

  def display(self, image_size): 
    """
    :param image_size: n where the image is nxn
    :returns: Two layer objects, one for the edges, one for the nodes
    """
    node_positions, node_layer = self.draw_node_layer(image_size)
    connection_layer = self.draw_connection_matrix(node_positions)

    return [connection_layer, node_layer]

  def draw_connection_matrix(self, node_positions):
    """
    :param node_positions: A 2D matrix of tuples (x, y) where (x, y) are positions of nodes drawn in the image s.t. 
    node_position[x][y] corresponds to
    posision on the boar self.board[x][y]
    :returns: A layer object containing alled ges to be drawn
    """
    # Note that connection_matrix already does not contain any connections between non-existing nodes, s.t. this code 
    # can work for both diamond and triangle shapes
    lines = []
    for row_num in range(0, len(self.connection_matrix)):
      row = self.connection_matrix[row_num] 
      for col_num in range(0, len(row)):
        if self.connection_matrix[row_num][col_num]:  
          # If there is a connection make a line
          # Remember that each entry in self.board contains a list of every connection to every other node, i.e it 
          # is n*n where n is len(self.board)
          from_row = row_num // len(self.board)
          from_col = row_num % len(self.board) 
          to_row = col_num // len(self.board) 
          to_col = col_num % len(self.board)

          from_position = node_positions[from_row][from_col]
          to_position = node_positions[to_row][to_col]
          # These values are now an index into self.board that finds the corresponding from and to nodes by using the 
          # connection matrix. Since node_positions has same format as self.board, we can use the same indexes there to 
          # find the start and to positions of the connection lines. 
          line = Line(from_position, to_position, (0, 0, 0))
          lines.append(line)
    layer = Layer(0, lines)
    return layer

  def draw_node_layer(self, image_size): 
    """
    :param image_size: n where image to be drawn is nxn 
    :returns: A layer object containing all the circles corresponding to nodes, a 2D matrix of postions in the 
    image s.t. each element in the matrix corresponds to the board
    position with the same index
    """
    max_width = len(self.board) 
    image_center = image_size/2.0
    side_padding = image_size/10.0
    content_width = image_size - 2*side_padding
    # Divide halv the content_width on the max number of steps
    step_size = content_width*0.5/(max_width-1)
    node_size = step_size/4

    node_positions = []
    circle_objects = []
    if self.shape == ShapeType.DIAMOND:
      # Calculate position of every node in image
      for row_num in range(len(self.board)): 
        row = self.board[row_num]
        node_position_row = [] 
        for col_num in range(len(row)):
          x_pos = image_center+(col_num-row_num)*step_size
          y_pos = side_padding+(col_num+row_num)*step_size
          node_position_row.append((x_pos, y_pos))
        node_positions.append(node_position_row)
      
      # Create circle objects
      for row_num in range(0, len(self.board)): 
        row = self.board[row_num]
        for col_num in range(0, len(row)):
          circle_obj = Circle(node_positions[row_num][col_num], node_size, row[col_num].get_color())
          circle_objects.append(circle_obj)

      node_layer = Layer(1, circle_objects)

    elif self.shape == ShapeType.TRIANGLE: 
      # Code is almost the same, just that it makes sure to skip nodes that aren't actually part of 
      # the matrix/are on wrong side of diagonal
      for row_num in range(len(self.board)):
        row = self.board[row_num]
        node_position_row = [] 
        for col_num in range(len(row)):
          if row[col_num] == False:
            # Just a sign to not create a display circle later
            node_position_row.append(False)
          else:
            x_pos = image_center+col_num*step_size-(row_num*step_size*0.5)
            y_pos = side_padding+row_num*step_size
            node_position_row.append((x_pos, y_pos))
        node_positions.append(node_position_row) 

      for row_num in range(0, len(self.board)):
        row = self.board[row_num]
        for col_num in range(0, len(row)):
          # If node_positions w/index has any value other than false
          if node_positions[row_num][col_num]:  
            circle_obj = Circle(node_positions[row_num][col_num], node_size, row[col_num].get_color())
            circle_objects.append(circle_obj)
      node_layer = Layer(1, circle_objects)

    return node_positions, node_layer

  def get_win(self):
    """
    :returns: True if win, False if not determined or not win
    """
    if self.count_remaining_pieces() == 1:
      return True
    else:
      # Note that false here merely means not won YET
      return False  
  
  def get_game_over(self):
    """
    :returns: True if game over (no more moves), False if not game over (more moves)
    """
    all_moves = self.get_all_moves() 
    return len(all_moves) == 0
  
  def count_remaining_pieces(self):
    """
    :returns: Number of the remaining pieces on the board.
    """
    remaining_pieces = 0
    for num0 in range(0, len(self.board)):
      for num1 in range(0, len(self.board[num0])):
        val = self.board[num0][num1]
        if val:  # if val is not false
          if val.state == PegState.UNSELECTED or val.state == PegState.SELECTED: 
            remaining_pieces += 1
    return remaining_pieces

  def get_empty_board_indecies(self):
    """
    Returns the positions for all empty board positions as (row, col)
    """
    empty_indecies = []
    for row_num in range(len(self.board)):
      for col_num in range(len(self.board)):
        if self.board[row_num][col_num] and self.board[row_num][col_num].state == PegState.REMOVED:
          empty_indecies.append((row_num, col_num))
    return empty_indecies
  
  def get_neighbours_and_directions(self, from_position):
    """
    Gets a list of all neighbours from a position towards all directions. 
    :param from_position: index of format (row, col) into self.board
    """
    
    # Transform index into board matrix into index into index into neighbour matrix
    from_row_index = self.board_to_connection_index(from_position)
    row = self.connection_matrix[from_row_index]
    
    neighbours = []
    for col_num in range(0, len(row)): 
      if row[col_num]:
        # Transform index into board index
        board_index = self.connection_to_board_index(col_num)
        if self.board[board_index[0]][board_index[1]].state != PegState.REMOVED:
          neighbours.append((board_index, row[col_num]))  # Store board index and direction in neighbours
    return neighbours

  def get_neighbour_in_direction(self, from_position, direction):
    """
    Gets a single neighbour from a position towards a specified direction. Returns False if
    there is no connection in the connection matrix in this direction.
    :param from_position: index of format (row, col) into self.board
    """
    # Transform index into board matrix into index into index into neighbour matrix
    from_row_index = self.board_to_connection_index(from_position)  
    row = self.connection_matrix[from_row_index]
    if direction in row:
      connection_row_index = row.index(direction)
      board_index = self.connection_to_board_index(connection_row_index)
      return board_index
    else:  
      False  # If direction is not in row. i.e. node is on an edge of the board

  def get_moves_to(self, node_position):
    """
    Gets legal moves to node_position. A move is legal if a neighbouring node from an empty node is a peg, and that 
    nodes neighbour in the same direction is also a peg. This function therefore assumes that the node_position is an 
    empty board position (i.e. PegState.REMOVED). This is also checked. 
    :param node_position: index of format (row, col) into self.board
    """
    if not self.board[node_position[0]][node_position[1]].state == PegState.REMOVED:
      raise IllegalArgumentException("node_position must be an index into the board matrix which is a peg with \
        PegState.REMOVED, i.e. an empty but legal board position.")
    
    moves = []

    # Find every neighbour of the empty node, and the direction of that neighbour
    neighbours_and_directions = self.get_neighbours_and_directions(node_position) 
    
    # For every neighbor, unpack it's position and direction
    for (position, direction) in neighbours_and_directions:

      # Then find it's neighbour in the same direction, if any.
      next_neighbour_in_direction = self.get_neighbour_in_direction(position, direction)   
      if next_neighbour_in_direction:
         # If there exists a neighbour in that direction, unpack it's position
        (next_neighbour_in_direction_x_pos, next_neighbour_in_direction_y_pos) = next_neighbour_in_direction

        peg_state = self.board[next_neighbour_in_direction_x_pos][next_neighbour_in_direction_y_pos].state
        
        if (peg_state == PegState.UNSELECTED or peg_state == PegState.SELECTED): 
          # If the neighbour is a peg that does not have PegState.REMOVED, i.e is a peg that is still present
          # Append a move from the node to the parameter node_position in a dictionary format
          moves.append({"from":next_neighbour_in_direction, "to":node_position})

    return moves
  
  def get_all_moves(self):
    """
    Gets all legal moves for the current board state. (self.board)
    """
    
    all_moves = []

    # Get all empty positions
    empty_node_indecies = self.get_empty_board_indecies()

    for node in empty_node_indecies:
      # Finds and stores all possible moves to the empty positions
      moves = self.get_moves_to(node)
      for move in moves:
        all_moves.append(move)
    
    return all_moves
  
  def board_to_connection_index(self, from_position):
    """
    :param from_position: A postion in (row, col) board position format
    :returns: index to a row in the connection matrix that corresponds to the node indexed in the 
    board matrix by from_position
    """
    # These are different due to the fact that the boards are rotated different ways to fit into a square matrix
    if self.shape == ShapeType.DIAMOND:
      return from_position[0] + from_position[1]*len(self.board)
    elif self.shape == ShapeType.TRIANGLE:
      return from_position[0]*len(self.board) + from_position[1]

  def connection_to_board_index(self, connection_index):
    """
    :param conection_index: A node represented as a index into a connection matrix row
    :returns: index to board matrix corresponding to index in a connection matrix row
    """
    # These are different due to the fact that the boards are rotated different ways to fit into a square matrix
    if self.shape == ShapeType.DIAMOND:
      board_index = (connection_index % len(self.board), connection_index // len(self.board))
      return board_index
    elif self.shape == ShapeType.TRIANGLE:
      board_index = (connection_index // len(self.board), connection_index % len(self.board))
      return board_index

  def do_action(self, action):
    # Set the last selected piece to not be displayed as last selected
    # Note that selection has no impact on game rules, but it is used for the visual display
    if not self.last_selected_piece == None:
      self.last_selected_piece.change_state(PegState.UNSELECTED)
    
    from_node = action['from']
    to_node = action['to']
    # Calculate distance between nodes
    middle_node = (to_node[0] - from_node[0], to_node[1] - from_node[1])

    # Use distance between nodes to calculate position of node in middle
    middle_node = (middle_node[0]//2 + from_node[0], middle_node[1]//2 + from_node[1])
    # print("From:", from_node, "To:", to_node, "Middle:", middle_node)

    # Find nodes from indexes
    from_node = self.board[from_node[0]][from_node[1]]
    middle_node = self.board[middle_node[0]][middle_node[1]]
    to_node = self.board[to_node[0]][to_node[1]]
    # Remove the moved and removed pegs
    from_node.remove_peg()
    middle_node.remove_peg()
    
    # Set to_node to be selected, and save the peg as selected
    to_node.change_state(PegState.SELECTED)
    self.last_selected_piece = to_node
  