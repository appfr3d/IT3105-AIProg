from PIL import Image, ImageDraw, ImageOps
# import matplotllib.pyplot as plt
from cv2 import cv2
import numpy as np

# "Interface" class for drawable entities
class DrawableInterface():
  def __init__(self):
    pass
  
  def draw(self, drawer):
    pass


class Circle(DrawableInterface):
  """
  A circle is a drawable object
  """
  def __init__(self, position, radius: int, color):
    """
    :param position: Tuple of two intergers, position in image
    :param color: Tuple of two integers in the range of [0, 255], color of circle
    """
    self.top_x = position[0]-radius
    self.top_y = position[1]-radius
    self.bottom_x = position[0] + radius
    self.bottom_y = position[1] + radius
    self.color = color
  
  def draw(self, drawer):
    """
    Draws the circle on an image
    :param drawer: An image drawing object
    """
    drawer.ellipse((self.top_x, self.top_y, self.bottom_x, self.bottom_y), fill=self.color)

class Line(DrawableInterface):
  """
  A drawable line
  """
  def __init__(self, from_position, to_position, color, width=3):
    """
    :param from_position: A tuple consiting of two integers (x, y) where (x, y) is coordinates of start point, counting from (0, 0) as upper left
    :param to_position: A tuple consiting of two integers (x, y) where (x, y) is the coordinate of the end point, counting from (0, 0) as upper left
    :param color: A tuple of three integers in the range [0, 255] signifying the RGB color of the line
    :param width: How wide the line should be, integer
    """
    (self.top_x, self.top_y) = from_position
    (self.bottom_x, self.bottom_y) = to_position
    self.color = color
    self.width = width
  
  def draw(self, drawer):
    drawer.line((self.top_x, self.top_y, self.bottom_x, self.bottom_y), fill=self.color, width=self.width)

class Layer(DrawableInterface):
  """
  A drawable object that acts as an umbrella for drawing other objects. Layers are used to organize the order of drawing other objects. 
  """
  def __init__(self, layer_position: int, data):
    """
    :param layer_position: The order the layer should be drawn in, where layers at the same position is drawn in some order. First drawn layer is
    at position 0
    :param data: A list of other drawable objects that will be drawn when draw is called on the layer. Other layers can be in this list, but
    their layer_position setting will not impact drawing order, they will be drawn in FIFO order in the data list. 
    """
    self.layer_position = layer_position
    self.data = data
  
  def draw(self, drawer):
    for drawable in self.data:
      drawable.draw(drawer)

class ImageDisplay():
  """
  A class that displays a  sim_world by drawing the layers it collects form its display function in correct order. 
  """
  def __init__(self, sim_world, image_size = 1000):
    """
    :param board: A SimWorld object supporting the display function
    :param image_size: n where the image size is nxn pixels
    """
    self.sim_world = sim_world
    self.image_size = image_size

  def display(self, frame_delay):
    """
    Generates image representing the sim world
    """
    layers = self.sim_world.display(self.image_size)
    layers.sort(key=lambda x: x.layer_position)
    img = Image.new('RGB', (self.image_size, self.image_size), color='white')
    
    
    drawer = ImageDraw.Draw(img)
    for layer in layers:
      layer.draw(drawer)

    img = ImageOps.flip(img)
    
    cv2.imshow('image',np.array(img))
    cv2.waitKey(frame_delay)

    # img.show()
    
      
