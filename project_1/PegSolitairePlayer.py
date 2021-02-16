from ConfigReader import ConfigReader
from pegSolitaireBoard import PegSolitaireBoard
from simWorld import SimWorldPlayer
import Pieces
from simWorld import ShapeType
from SimWorldDisplayer import ImageDisplay


class PegSolitairePlayer(SimWorldPlayer):
  """
  Bridge between the sim-world and the reinforcement agent
  """

  def __init__(self, board_empty_positions, board_shape, board_size, image_size, frame_delay, win_reward, base_reward, peg_loss, peg_loss2, moves_loss):
    # Saves board params for re-initialization
    self.board_empty_positions = board_empty_positions
    self.board_shape = board_shape
    self.board_size = board_size
    self.game = PegSolitaireBoard(board_empty_positions, board_shape, board_size)
    self.sim_world_displayer = ImageDisplay(self.game, image_size)
    self.display = False
    self.image_size = image_size
    self.frame_delay = frame_delay
    self.win_reward= win_reward
    self.base_reward = base_reward
    self.peg_loss = peg_loss
    self.peg_loss2 = peg_loss2
    self.moves_loss = moves_loss

  def reset_state(self):
    # Re-initialize board
    self.game = PegSolitaireBoard(self.board_empty_positions, self.board_shape, self.board_size)
    self.sim_world_displayer = ImageDisplay(self.game, self.image_size)

  def get_state(self):
    state = []
    for i0 in range(len(self.game.board)):
      for i1 in range(len(self.game.board[0])):
        if self.game.board[i0][i1] != False:  # If this is not a "non-location" (Triangle board)
          if self.game.board[i0][i1].state == Pieces.PegState.SELECTED or self.game.board[i0][
            i1].state == Pieces.PegState.UNSELECTED:
            state.append(1)
          else:
            state.append(0)
        else:
          state.append(0)
    return tuple(state)

  def get_actions(self):
    moves = self.game.get_all_moves()
    rows = len(self.game.board)
    actions = []
    for move in moves:
      from_node = move["from"]
      to_node = move["to"]
      # Maps from 2d board rep to 1d board rep
      from_index = from_node[0] * rows + from_node[1]
      to_index = to_node[0] * rows + to_node[1]
      actions.append((from_index, to_index))

    return actions

  def do_action(self, action):
    """
    Do a RL action on the board
    :param action: A tuple of two numbers, the from state and the to state
    """

    rows, columns = len(self.game.board), len(self.game.board[0])
    # Maps from 1d board to 2d board
    from_node = (action[0] // columns, action[0] % rows)
    to_node = (action[1] // columns, action[1] % rows)
    action = {'from': from_node, 'to': to_node}
    self.game.do_action(action)

    if self.display:
      self.sim_world_displayer.display(self.frame_delay)

  def get_game_over(self):
    return self.game.get_game_over() 

  def get_reward(self):
    if self.game.get_game_over() and self.game.get_win():
      return self.win_reward
    
    num_pegs_remaining = float(self.get_remaining_pegs())
    # Remaining peg heuristic: Amount of remaining begs, scaled by board size
    remaining_peg_heuristic = (float(self.board_size*self.board_size)-num_pegs_remaining)/(float(self.board_size*self.board_size))

    # Available moves heuristic: Remaining moves/remaining pegs. A state with many possible moves per peg is more likely to have a successor state that is a victory state (because there are more moves)
    available_moves_heuristic = len(self.game.get_all_moves())/num_pegs_remaining
    
    return self.base_reward - self.peg_loss * num_pegs_remaining - self.peg_loss2 * remaining_peg_heuristic + available_moves_heuristic * self.moves_loss
    # #if self.game.get_game_over() and self.game.get_win():
    # #  return 1
    # return 0

  def get_remaining_pegs(self):
    return self.game.count_remaining_pieces()

  def force_display_frame(self):
    # Make the sim_world_displayer display a frame
    self.sim_world_displayer.display(self.frame_delay)
