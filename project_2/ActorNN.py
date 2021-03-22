from tensorflow import keras
import tensorflow as tf
from PlayerEnum import Player
from Pieces import PegState
import numpy as np
import time

class ActorNN:
  def __init__(self, config, game_bridge):
    """
    :param config: Config file interpreter object
    :param game_bridge: A subclass of the generic GameBridge class that translates game representation into appropriate form for NN
    """
    self.config =  config
    self.game_bridge = game_bridge
    self.model = self.generate_network()
    self.epsilon = config.initial_epsilon
    # k^2 output nodes
    # normalize output nodes to only actions that are valid (k^2 - q)
    # 2 * config.size^2 input nodes + 2 
    # format: player1TurnBit Player2TurnBit Position1OwnedByPlayer1Bit Position1OwnedByPlayer2Bit etc...
    # 00 ikke eid
    # 01 player 1
    # 10 player 2
    # hashed game state as input
  
  def generate_network(self):
    opt = self.config.optimizer  # Stocastic Gradient Decent
    model = keras.models.Sequential()
    loss = self.game_bridge.get_loss_metric()

    input_shape = self.game_bridge.get_input_shape()

    model.add(keras.Input(shape=input_shape))

    # Add fully connected layers to the model
    for dim in self.config.neurons_per_layer: 
      model.add(keras.layers.Dense(dim, activation=self.config.activation_func))

    model.add(self.game_bridge.get_output_layer())

    model.compile(optimizer=opt(lr=self.config.actor_learning_rate), loss=loss, metrics=[loss])
    model.build(input_shape = self.game_bridge.get_input_shape())
    return model

  def eval(self, param_tuple):
    # What is called by monte-carlo tree node
    inputs = self.game_bridge.translate_to_nn_input(param_tuple)
    
    output = self.model(inputs).numpy()

    return self.game_bridge.post_process(output, param_tuple)
  
  def fit(self, RBUF):
    # dict with labels 'x', 'y'
    training_samples_dict = self.game_bridge.process_training_samples(RBUF)
    x = training_samples_dict['x']
    y = training_samples_dict['y']

    self.model.fit(x=x, y=y)
  
  def save(self):
    self.model.save('/tournament_models/model_' + str(time.time()))

  def load(self, path):
    self.model = keras.load(path)


class GameBridge:
  def __init__(self):
    pass
  
  def process_training_samples(self, RBUF):
    # Get random minibatch from RBUF in list format with (input, target) tuples of appropriate format for NN 
    pass
  
  def get_input_shape(self):
    # Get the appropriate input shape of the NN for this game
    pass

  def get_output_layer(self):
    # Get the appropriate output layer for this game type
    pass
  
  def get_loss_metric(self):
    # Get the appropriate loss metric for this game type
    pass
  
  def translate_to_nn_input(self, params): 
    # Translate input data about board state to a format appropriate is input to the NN for evaluation
    pass

  def post_process(self, nn_output, params):
    # If necessary you can post-process network output (when predicting, not when training) before it is returned
    pass


class HexBoardNNBridge(GameBridge):
  def __init__(self, config):
    self.config = config
  
  def process_training_samples(self, RBUF):
    # Get random minibatch of RBUF
    # RBUF = [((board, player_to_move), distribution)]
    # training_samples = [(input, target)]
    # input = [0, 1, 0 , 0, ... ] binary version of board state
    # target is just distribution
    
    
    def map_to_sample(RBUF_pair):
      board = RBUF_pair[0][0]
      moves = board.get_all_moves()
      player_to_move = RBUF_pair[0][1]

      binary_input = self.translate_to_nn_input((moves, player_to_move, board.board))
      distribution = RBUF_pair[1]

      return (binary_input, distribution)
    
    # For now, just return every case
    training_samples = [map_to_sample(RBUF_pair) for RBUF_pair in RBUF]
    x = np.zeros((len(training_samples), len(training_samples[0][0][0])))
    y = np.zeros((len(training_samples), len(training_samples[0][1])))

    for num in range(len(training_samples)):
      x[num] = training_samples[num][0][0]
      y[num] = training_samples[num][1]
    
    return {'x':x, 'y':y}
  
  def get_input_shape(self): 
    return (2 + 2*self.config.size*self.config.size,)
  
  def get_output_layer(self):
    return keras.layers.Dense(self.config.size*self.config.size, activation=tf.nn.softmax)
  
  def get_loss_metric(self):
    return keras.losses.categorical_crossentropy
  
  def translate_to_nn_input(self, params):
    moves = params[0] # hex_board.get_all_moves()
    player_to_move = params[1]
    board_representation = params[2]

    inputs = np.zeros(shape=(1, 2 + 2 * self.config.size*self.config.size))
    index = 0
    if player_to_move == Player.PLAYER1:
      inputs[0, index] = 0
      index += 1
      inputs[0, index] = 1
      index += 1
    else:
      inputs[0, index] = 1
      index += 1
      inputs[0, index] = 0
      index += 1
    
    for row in range(len(board_representation)):
      for col in range(len(board_representation[0])):
        val = board_representation[row][col]
        if val.state == PegState.PLAYER1:
          inputs[0, index] = 0
          index += 1
          inputs[0, index] = 1
          index += 1
        elif val.state == PegState.PLAYER2:
          inputs[0, index] = 1
          index += 1
          inputs[0, index] = 0
          index += 1
        else: 
          inputs[0, index] = 0
          index += 1
          inputs[0, index] = 0
          index += 1
    
    return inputs
  
  def post_process(self, nn_output, params): 
    nn_output = nn_output.reshape((16,))
    moves = params[0]
    player_to_move = params[1]
    board_representation = params[2]
    legal_moves = moves # 2D array with 1 for legal move 0 for not

    # Mask out non-legal moves
    values = self.mask(nn_output, np.asarray(legal_moves).flatten())

    # Normalize and return
    # norm = np.linalg.norm(values) # Normalization is not necessary for just picking largest value as action choice

    index = list(values).index(max(values))
    pos = (index // self.config.size, index % self.config.size)

    return (pos, player_to_move) # ((x, y), player_to_move)

  def mask(self, values, mask_values):
    """
    Map values to 0 if mask_values at elementwise same position is 0, or to original value of mask_values is 1
    """
    return np.multiply(values, mask_values)

  
    