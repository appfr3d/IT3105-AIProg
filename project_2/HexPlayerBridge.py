
class HexPlayerBridge:
  def __init__(self, hex_board):
    self.hex_board = hex_board

  def get_actions(self):
    pass

  def get_state(self):
    pass

  def get_win(self):
    return self.hex_board.get_win()


