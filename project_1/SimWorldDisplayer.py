# "Interface" class for drawable entities
class DrawableInterface():
  def __init__(self):
    pass
  
  def draw(self, drawer):
    pass


class Circle(DrawableInterface):
  def __init__(self, position: tuple(int, int), radius: int, color: tuple(int, int, int)):
    # transform position and radius into start and stop points of a box containing the circle. The topmost and leftmost pixel in an image is (0, 0).
    self.top_x = position[0]-radius
    self.top_y = position[1]-radius
    self.bottom_x = position[0] + radius
    self.bottom_y = position[1] + radius
    self.color = color
  
  def draw(self, drawer):
    drawer.ellipse((self.top_x, self.top_y, self.bottom_x, self.bottom_y), fill=self.color)

class Line(DrawableInterface):
  def __init__(self, from_position: tuple(int, int), to_position: tuple(int, int), color: tuple(int, int, int), width: int):
    (self.top_x, self.top_y) = from_position
    (self.bottom_x, self.bottom_y) = to_position
    self.color = color
    self.width = width
  
  def draw(self, drawer):
    drawer.line((self.top_x, self.top_y, self.bottom_x, self.bottom_y), fill=self.color, width=self.width)

class Layer(DrawableInterface):
  def __init__(self, layer_position: int, data):
    self.layer_position = layer_position
    self.data = data
  
  def draw(self, drawer):
    for drawable in self.data:
      drawable.draw(drawer)


