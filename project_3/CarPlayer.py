from SimWorld import SimWorldPlayer
from CarWorld import CarWorld
from SimWorldDisplayer import ImageDisplay
import numpy as np
import math

class CarPlayer(SimWorldPlayer):
  def __init__(self, config):
    self.config = config
    self.state = CarWorld(self.config)
    self.sim_world_displayer = ImageDisplay(self.state, self.config.image_size)
    self.display = False
    tiler = TwoDimTileFactory(-1.2, 0.6, -0.7, 0.7, self.config)
    self.tiles = tiler.make_tiles()
    self.has_won = False
    #self.seen_tilings = []
    
  def reset_state(self):
    self.state = CarWorld(self.config)
    self.sim_world_displayer = ImageDisplay(self.state, self.config.image_size)

  def process_x_vel(self, x_pos, velocity):
    tiles_per_tile = self.tiles[0].get_tile_count()
    states = []
    for num in range(3):
      nn_input = np.zeros((3, len(self.tiles), tiles_per_tile))
      for num2 in range(len(self.tiles)):
        x, y = self.tiles[num2].get_tile(x_pos, velocity)
        indx = y * self.tiles[num2].y_tiles + x
        # print(tiles_per_tile*num)
        # print(x)
        # print(x*self.tiles[num].y_tiles)
        # print(y)
        nn_input[num, num2, indx] = 1
      states.append(nn_input.flatten().reshape(1, 3 * len(self.tiles) * tiles_per_tile))
    return states

  def get_state(self, append=True):
    x_pos = self.state.state['x-pos']
    velocity = self.state.state['velocity']

    state = self.process_x_vel(x_pos, velocity)
    #if append:
    #  self.seen_tilings.append(str(state))
    return state

  def get_actions(self):
    return self.state.get_actions()

  def do_action(self, action):
    # map from nn output to force
    action = float(action)-1
    # then do action
    #print(action, end=',    ')
    self.state.make_move({'force': action})
    if self.display:
      self.sim_world_displayer.display(self.config.frame_delay)

  def get_game_over(self):
    return self.state.get_game_over()
  
  def get_reward(self):
    # def get_y_pos(x_pos):
    #   return math.cos(3*(x_pos + math.pi/2))

    if self.state.get_win():
      self.has_won = True
      return self.config.win_reward

    #if str(self.get_state(append=False)) in self.seen_tilings:
    #  return self.config.base_reward
    return self.config.base_reward

  def get_log_metric(self):
    return self.state.get_log_metric()

  def force_display_frame(self):
    # Make the sim_world_displayer display a frame
    self.sim_world_displayer.display(self.config.frame_delay)

  # def display_tiles(self):
  #   import matplotlib.pyplot as plt
  #   x_bounds = [-1.2, 0.6]
  #   y_bounds = [-0.07, 0.07]
  #   for t_i in range(len(self.tiles)):
  #     tile = self.tiles[t_i]
  #     x = []
  #     y = []
  #     for 



class TwoDimTileFactory:
  def __init__(self, dim1min, dim1max, dim2min, dim2max, config):
    # Note: Not necessarily perfect rectangles
    self.dim1min = dim1min
    self.dim1max = dim1max
    self.dim2min = dim2min
    self.dim2max = dim2max
    self.config = config
    self.tiles_per_tile_sqrt = config.tiles_per_tile
    self.x_interval = (dim1max-dim1min)/(self.tiles_per_tile_sqrt)
    self.y_interval = (dim2max-dim2min)/(self.tiles_per_tile_sqrt)
  
  def make_tiles(self):
    tiles = []
    for num in range(self.config.tiles):
      # Odd numbers per reccomendation in Sutton \
      # alpha = self.config.tiling_offset/(self.config.tiles)
      x_offset = num * (self.x_interval/self.config.tiles)
      y_offset = num * (self.y_interval/self.config.tiles)
      tile = TwoDimTile(self.x_interval, self.y_interval, x_offset, y_offset, self.tiles_per_tile_sqrt, self.tiles_per_tile_sqrt, self.dim1min, self.dim2min)
      tiles.append(tile)
    return tiles


class TwoDimTile:
  def __init__(self, x_interval, y_interval, x_offset, y_offset, x_tiles, y_tiles, x_min, y_min):
    self.x_interval = x_interval
    self.y_interval = y_interval
    self.x_offset = x_offset
    self.y_offset = y_offset
    self.x_tiles = x_tiles
    self.y_tiles = y_tiles
    self.x_min = x_min
    self.y_min = y_min

  def get_tile_count(self):
    return self.x_tiles * self.y_tiles

  def get_tile(self, x, y):
    x0 = x - self.x_min + self.x_offset
    y0 = y - self.y_min + self.y_offset
    x_tile = int(x0 // self.x_interval)
    y_tile = int(y0 // self.y_interval)
    if (x_tile >= self.x_tiles):
      x_tile = self.x_tiles-1
    if (y_tile >= self.y_tiles):
      y_tile = self.y_tiles-1
    return x_tile, y_tile