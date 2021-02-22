from simWorld import HexBoard
from simWorld import ShapeType
from Pieces import Peg, PegState
from SimWorldDisplayer import Line, Circle, Layer
from IllegalArgumentException import * 
from PlayerEnum import Player
class HexGameBoard(HexBoard):

  def __init__(self, shape, size):
    """
    :param empty_position: A list of positions, counted as shown in /docs/Diamond_Connection_Matrix.png and
    /docs/Triangle_Connection_Matrix.png
    :param shape: A ShapeType
    :param size: How wide the node matrix should be across it's diagonal
    """
    super(HexGameBoard, self).__init__([num for num in range(0, size*size)], shape, size)
    self.init_board_pieces()

  def init_board_pieces(self):
    """
    Initialize the board 
    """
    if self.shape == ShapeType.DIAMOND:
      for position in self.empty_positions: 
        row = position//self.size
        col = position%self.size
        self.board[row][col].change_state(PegState.EMPTY)

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


    return node_positions, node_layer

  def get_win(self):
    # TODO:
    # If connection from left column to right column with Player1 pieces return Player.Player1
    # if from top row to bottom row with player2 pieces return Player.Player2
    # else False
    return False

  def count_empty_positions(self):
    """
    :returns: Number of the remaining empty piece positions on the board.
    """
    empty_positions = 0
    for num0 in range(0, len(self.board)):
      for num1 in range(0, len(self.board[num0])):
        val = self.board[num0][num1]
        if val:  # if val is not false
          if val.state == PegState.EMPTY:
            empty_positions += 1
    return empty_positions

  def get_empty_board_indecies(self):
    """
    Returns the positions for all empty board positions as (row, col)
    """
    empty_indecies = []
    for row_num in range(len(self.board)):
      for col_num in range(len(self.board)):
        if self.board[row_num][col_num] and self.board[row_num][col_num].state == PegState.EMPTY:
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
        if self.board[board_index[0]][board_index[1]].state != PegState.EMPTY:
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

  def get_all_moves(self):
    """
    Gets all legal moves for the current board state. (self.board). Returns a matrix of True/False where True means
    that it is legal to place there
    """
    
    all_moves = []
    for row in self.board:
      move_row = []
      for col in row:
        move_row.append(True if col.PegState == PegState.EMPTY else False)
      all_moves.append(move_row)
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

  def connection_to_board_index(self, connection_index):
    """
    :param conection_index: A node represented as a index into a connection matrix row
    :returns: index to board matrix corresponding to index in a connection matrix row
    """
    # These are different due to the fact that the boards are rotated different ways to fit into a square matrix
    if self.shape == ShapeType.DIAMOND:
      board_index = (connection_index % len(self.board), connection_index // len(self.board))
      return board_index

  def do_action(self, action):
    pos, player = action
    if player == Player.PLAYER1:
      self.board[pos[0]][pos[1]].player1_place_peg()
    else:  # Player2
      self.board[pos[0]][pos[1]].player2_place_peg()
