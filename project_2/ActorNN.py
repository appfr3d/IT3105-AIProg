from tensorflow import keras
import tensorflow as tf
from PlayerEnum import Player
from Pieces import PegState
import numpy as np
import time
import random
import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# These are the crossentropy functions that Keith sent on mail.
# As this is clearly the most reasonable implementation (because it both uses
# tensorflow functions in computation and because it is "safe" from predictions close to 0)
# we will just use it too, assume this is fine as it was provided by Keith
def deepnet_cross_entropy(targets, outs):

  return tf.reduce_mean(tf.reduce_sum(-1 * targets * safelog(outs), axis=[1]))

  # The use of mean here is because I’m sending in minibatches of targets and outputs.


def safelog(tensor, base=0.0001):
  return tf.math.log(tf.math.maximum(tensor, base))


class ActorNN:
  def __init__(self, config, game_bridge, model_path=None):
    """
    :param config: Config file interpreter object
    :param game_bridge: A subclass of the generic GameBridge class that translates game representation into appropriate form for NN
    """
    self.config =  config
    self.game_bridge = game_bridge

    if model_path == None:
      self.model = self.generate_network()
    else:
      self.load(model_path)
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
    # Consider conv2d if regular don't work
    pre_layers = self.game_bridge.get_pre_layers()
    for layer in pre_layers:
      model.add(layer)
    for dim in self.config.neurons_per_layer: 
      model.add(keras.layers.Dense(dim, activation=self.config.activation_func, kernel_regularizer=keras.regularizers.l2()))
      #model.add(keras.layers.BatchNormalization())
      # Batch normalization layers makes the output of each neuron more like a normal gaussian.
      # This often helps w/training time and generalization.

    # Consider sigmoid here?
    output_layers = self.game_bridge.get_output_layer()
    for layer in output_layers:
      model.add(layer)

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

    self.model.fit(x=x, y=y, verbose=0, epochs=self.config.epochs_per_rbuf)

  def fit_no_processing(self, RBUF):
    x = RBUF['x']
    y = RBUF['y']

    self.model.fit(x=x, y=y, verbose=0, epochs=self.config.epochs_per_rbuf)
  
  def save(self, save_path, episode):
    if save_path == None:
      model_path = CURRENT_DIR + '/tournament_models/' + str(self.config.size) + '/model_' + str(episode).zfill(8)
      self.model.save(model_path)
    # print('\nSaving model to: ' + model_path + '\n')
    else:
      self.model.save(save_path + '/' + str(episode).zfill(8))
 
  def load(self, path):
    self.model = keras.models.load_model(path, compile=False)

    
    if self.config.run_type == 'train_again':
      # Compile and build model for further training.
      opt = self.config.optimizer
      loss = self.game_bridge.get_loss_metric()
      self.model.compile(optimizer=opt(lr=self.config.actor_learning_rate, clipnorm=1.0), loss=loss, metrics=[loss])
      self.model.build(input_shape = self.game_bridge.get_input_shape())


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

  def get_pre_Layers(self):
    # Get layers which should come directly after input layer, ex. convolutional layers or other layers which are not
    # simple fully connected layers that can be set in the config file
    pass

  def get_output_size(self):
    pass

  def get_input_size(self):
    pass

class HexBoardNNBridge(GameBridge):
  def __init__(self, config):
    self.config = config

  def get_output_size(self):
    return self.config.size*self.config.size

  def get_pre_layers(self):
    pre_layers = []
    size = self.config.size
    q = size
    y = 16
    if size != 6:
      new_shape = (size + 1) * (size + 1)
      # Map input which has data about which player and the board onto a larger list, which can be reshaped into a
      # board shape for use with conv2d
      pre_layers.append(keras.layers.Dense(new_shape, activation=self.config.activation_func,
                                           kernel_regularizer=keras.regularizers.l2(l2=1e-4),
                                           bias_regularizer=keras.regularizers.l2(1e-4)))
      # pre_layers.append(keras.layers.BatchNormalization())

      pre_layers.append(keras.layers.Reshape((size + 1, size + 1, 1,)))
      while q >= 4:
        pre_layers.append(
          keras.layers.Conv2D(filters=y, kernel_size=(2, 2), strides=2, activation=self.config.activation_func,
                              padding='same', kernel_regularizer=keras.regularizers.l2(l2=1e-4),
                              bias_regularizer=keras.regularizers.l2(1e-4)))
        # pre_layers.append(keras.layers.BatchNormalization())
        q = q / 2
        y *= 2
      pre_layers.append(keras.layers.Flatten())
    else:
      new_shape = (2*size + 1) * (2*size + 1)
      pre_layers.append(keras.layers.Dense(new_shape, activation=self.config.activation_func))
      pre_layers.append(keras.layers.Reshape((2*size + 1, 2*size + 1, 1,)))
      pre_layers.append(
        keras.layers.Conv2D(512, kernel_size=(5, 5), strides=2, activation=self.config.activation_func, padding='same', kernel_regularizer=keras.regularizers.l2()))
      pre_layers.append(
        keras.layers.Conv2D(512, kernel_size=(5, 5), strides=2, activation=self.config.activation_func, padding='same', kernel_regularizer=keras.regularizers.l2()))
      pre_layers.append(
        keras.layers.Conv2D(256, kernel_size=(2, 2), strides=1, activation=self.config.activation_func, padding='same', kernel_regularizer=keras.regularizers.l2()))
      pre_layers.append(
        keras.layers.Conv2D(128, kernel_size=(2, 2), strides=1, activation=self.config.activation_func,
                            padding='same', kernel_regularizer=keras.regularizers.l2()))
      pre_layers.append(
        keras.layers.Conv2D(64, kernel_size=(2, 2), strides=1, activation=self.config.activation_func,
                            padding='same', kernel_regularizer=keras.regularizers.l2()))
      pre_layers.append(
        keras.layers.Conv2D(64, kernel_size=(2, 2), strides=2, activation=self.config.activation_func, padding='same', kernel_regularizer=keras.regularizers.l2()))
      pre_layers.append(keras.layers.Flatten())
    return pre_layers

  def map_to_sample(self, RBUF_pair):
    board = RBUF_pair[0][0]
    moves = board.get_all_moves()
    player_to_move = RBUF_pair[0][1]

    binary_input = self.translate_to_nn_input((moves, player_to_move, board.board))
    distribution = RBUF_pair[1]

    return np.append(binary_input, distribution)

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

  def get_input_size(self):
    return 2 + 2*self.config.size*self.config.size

  def get_output_layer(self):
    return [keras.layers.Dense(self.config.size * self.config.size, activation='softmax')]
  
  def get_loss_metric(self):
    return lambda target, out: deepnet_cross_entropy(target, out)

  
  def translate_to_nn_input(self, params):
    moves = params[0] # hex_board.get_all_moves()
    player_to_move = params[1]
    board_representation = params[2]

    inputs = np.zeros(shape=(1, 2 + 2 * self.config.size*self.config.size))
    index = 0
    if player_to_move == Player.PLAYER1:
      # 01 means that player1 can place peg at a given position
      inputs[0, index] = 0
      index += 1
      inputs[0, index] = 1
      index += 1
    else:
      # 10 means that player2 can place peg at a given position
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
    nn_output = nn_output.reshape((self.config.size*self.config.size,))
    moves = params[0]
    player_to_move = params[1]
    board_representation = params[2]
    move_type = params[3]
    legal_moves = np.asarray(moves).flatten() # 2D array with True for legal move False for not
    legal_moves[legal_moves == True] = 1
    legal_moves[legal_moves == False] = 0

    # Mask out non-legal moves
    # crucially it just sets invalid 
    values = self.mask(nn_output, legal_moves)

    
    if move_type == 'greedy':
      while True: # Loop to fix issue with picking illegal moves as best move
        index = list(values).index(max(values))
        if legal_moves[index] == 1: # if legal move
          break
        else:
          values[index] = -1
    elif move_type == 'stochastic' or move_type == 'stochastic_pow':

      if move_type == 'stochastic_pow':
        # Make every number be 1 + num to avoid nan problems
        for indx in range(len(values)):
          values[indx] += 1
        # Increase probability of higher numbers by raising them to the power of 2
        for indx in range(len(values)):
          values[indx] = values[indx]**2

      # normalize
      the_sum = np.sum(values)

      if the_sum == 0:  # If the nn predicts very low values for every move that is legal the sum can become 0. If this happens, return the first legal move
        index = 0
        moves = np.asarray(legal_moves).flatten()
        while True:
          if legal_moves[index] == True or legal_moves[index] == 1:
            pos = (index // self.config.size, index % self.config.size)
            return (pos, player_to_move)
          index += 1

      for indx in range(len(values)):
        values[indx] = values[indx]/the_sum


      values = np.insert(values, 0, [0.0])

      # aggregate
      for indx in range(1, len(values)):
        values[indx] = values[indx] + values[indx-1]

      randval = random.random()
      index = 1

      # Choose by finding which interval randval is in, stochastic choice
      while True:
        if index == 37:
          print('what')
        if randval >= values[index-1] and randval < values[index] and values[index-1] != values[index]:
          # If it is in interval
          # And the values are not equal <=> it is not one of the invalid moves that were masked out to 0
          break
        else:
          index += 1
      index -= 1 # because we added a 0 at start of the values list
    elif move_type == 'e-greedy':
      e = self.config.initial_epsilon # FOR NOW assumes constant epsilon
      randval = random.random()
      if randval < e:
        index = random.randint(0, values.shape[0]-1)
      else:
        index = list(values).index(max(values))
    elif move_type == 'first-random-greedy':
      # If first move in the game, choose randomly
      # Else, choose greedy
      num_possible_moves = np.sum(np.asarray(legal_moves, dtype=np.bool))
      if num_possible_moves == self.config.size**2:
        index = random.randint(0, values.shape[0]-1)
      else:
        if max(values) != 0:
          index = list(values).index(max(values))
        else:
          index = 0
          while True:
            if legal_moves[index] == True or legal_moves[index] == 1:
              break
            index += 1

    pos = (index // self.config.size, index % self.config.size)

    return (pos, player_to_move) # ((x, y), player_to_move)

  def mask(self, values, mask_values):
    """
    Map values to 0 if mask_values at elementwise same position is 0, or to original value of mask_values is 1
    """
    return np.multiply(values, mask_values)

class HexBoardNNBridgeOnlineTournament(GameBridge):
  def __init__(self, config):
    self.config = config

  def get_pre_layers(self):
    pre_layers = []
    size = self.config.size
    q = size
    y = 16
    if size != 6:
      new_shape = (size+1)*(size+1)
      # Map input which has data about which player and the board onto a larger list, which can be reshaped into a
      # board shape for use with conv2d
      pre_layers.append(keras.layers.Dense(new_shape, activation=self.config.activation_func, kernel_regularizer = keras.regularizers.l2(l2=1e-4),
          bias_regularizer = keras.regularizers.l2(1e-4)))
      #pre_layers.append(keras.layers.BatchNormalization())

      pre_layers.append(keras.layers.Reshape((size+1, size+1, 1,)))
      while q >= 4:
        pre_layers.append(keras.layers.Conv2D(filters=y, kernel_size=(2, 2), strides=2, activation=self.config.activation_func, padding='same', kernel_regularizer = keras.regularizers.l2(l2=1e-4),
              bias_regularizer = keras.regularizers.l2(1e-4)))
        #pre_layers.append(keras.layers.BatchNormalization())
        q = q / 2
        y *= 2
      pre_layers.append(keras.layers.Flatten())
    else:
      new_shape = (size + 1) * (size + 1)
      pre_layers.append(keras.layers.Dense(new_shape, activation=self.config.activation_func))
      pre_layers.append(keras.layers.Reshape((size + 1, size + 1, 1,)))
      pre_layers.append(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=1, activation=self.config.activation_func, padding='same'))
      pre_layers.append(keras.layers.Conv2D(16, kernel_size=(5, 5), strides=2, activation=self.config.activation_func, padding='same'))
      pre_layers.append(
        keras.layers.Conv2D(8, kernel_size=(2, 2), strides=1, activation=self.config.activation_func, padding='same'))
      pre_layers.append(
        keras.layers.Conv2D(8, kernel_size=(2, 2), strides=2, activation=self.config.activation_func, padding='same'))
      pre_layers.append(keras.layers.Flatten())
    return pre_layers

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
    return [keras.layers.Dense(self.config.size*self.config.size, activation='softmax')]
  
  def get_loss_metric(self):
    return keras.losses.KLD

  
  def translate_to_nn_input(self, params):
    moves = params
    rep = []

    for datapoint in moves:
      if datapoint == 0:
        rep.append(0)
        rep.append(0)
      elif datapoint == 1:
        rep.append(0)
        rep.append(1)
      elif datapoint == 2:
        rep.append(1)
        rep.append(0)
    
    return np.asarray(rep).reshape(1, len(rep))
  
  def post_process(self, nn_output, params): 
    nn_output = nn_output.reshape((self.config.size*self.config.size,))
    moves = np.asarray(params[1:])  # First one is who gets to move
    # flip moves s.t. 1 indicates empty, 0 indicates not empty
    moves[moves == 0] = 10
    moves[moves == 1] = 0
    moves[moves == 2] = 0
    moves[moves == 10] = 1

    # Mask out non-legal moves
    # crucially it just sets invalid 
    values = self.mask(nn_output, np.asarray(moves).flatten())


    # Allways choose greedy in tournament
    index = list(values).index(max(values))
    pos = (index // self.config.size, index % self.config.size)

    return pos # ((x, y), player_to_move)

  def mask(self, values, mask_values):
    """
    Map values to 0 if mask_values at elementwise same position is 0, or to original value of mask_values is 1
    """
    return np.multiply(values, mask_values)



    