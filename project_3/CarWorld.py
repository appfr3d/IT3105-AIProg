from SimWorld import SimWorldBase
from SimWorldDisplayer import Layer, Line, Circle
import random
import math

class CarWorld(SimWorldBase):
  """
  A semi-abstract SimWorld type, used as a super-class of concrete implementations of Hexagonal game boards. 
  """
  def __init__(self, config):
    self.config = config
    self.move_count = 0
    
    # From assignment:
    # In the classic version of the problem, each episode begins with x randomly chosen in the range [-0.6, -0.4] 
    # and velocity initialized to zero.
    self.state = {'x-pos': random.uniform(-0.6, -0.4), 'velocity': 0}
  
  def get_win(self):
    return self.state['x-pos'] >= 0.6

  def get_game_over(self):
    return self.move_count == 1000 or self.get_win()

  def get_actions(self):
    return [-1, 0, 1]

  def make_move(self, move):
    pos = self.state['x-pos']
    vel = self.state['velocity']

    new_vel = max(min(vel + 0.001 * move['force'] - 0.0025*math.cos(3*pos), 0.07), -0.07)
    self.state['velocity'] = new_vel

    self.state['x-pos'] = max(min(pos + new_vel, 0.6), -1.2)

    self.move_count += 1
  
  def get_log_metric(self):
    return self.move_count

  def display(self, image_size):

    def get_y_pos(x_pos):
      return math.cos(3*(x_pos + math.pi/2))

    # Mountain
    lines = []
    line_res = 100
    padding = 100
    x_inc = (0.6 - (-1.2))/line_res
    
    last_x = x_inc * line_res
    res_mult_x = (image_size-(padding*2))/last_x
    res_mult_y = (image_size-(padding*2))/2 # cos has range of two

    for num in range(1, line_res):
      x0 = padding + (num-1) * x_inc * res_mult_x
      x1 = padding + num * x_inc * res_mult_x
      real_x0 = -1.2 + (num-1)*x_inc
      real_x1 = -1.2 + num * x_inc
      y0 = 5*padding + get_y_pos(real_x0) * res_mult_y
      y1 = 5*padding + get_y_pos(real_x1) * res_mult_y
      line = Line((x0, y0), (x1, y1), (0, 0, 0))
      lines.append(line)
      
    
    line_layer = Layer(0, lines)
    
    # Car
    x_pos = padding + ((self.state['x-pos']+1.2)/x_inc)*x_inc * res_mult_x
    y_pos = 5*padding + get_y_pos(self.state['x-pos']) * res_mult_y
    car = Circle((x_pos, y_pos), 10, (100, 0, 100))
    car_layer = Layer(1, [car])    

    return [line_layer, car_layer]

    