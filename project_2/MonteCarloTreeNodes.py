from Pieces import PegState
from HexGameBoard import HexGameBoard
from PlayerEnum import Player
import math
import numpy as np
import random
from Pieces import Peg

class TreeState:
  def __init__(self, game_bridge):
    self.tree_states = {}
    self.hasher = game_bridge

  def exists(self, state):
    return self.state_hash(state) in self.tree_states

  def get_existing_parents(self, state):
    hashed_state = self.state_hash(state)
    if self.state_hash(state) in self.tree_states:
      return self.tree_states[hashed_state].parents
    else:
      return []

  def state_hash(self, state):
    return self.hasher.hash(state)

  def add_node(self, node):
    # Assume node is a monte carlo tree node 
    self.tree_states[self.state_hash(node.state)] = node

  def get_node(self, state):
    return self.tree_states[self.state_hash(state)]

class GameBridge():
  def __init__(self, config):
    self.config = config

  def initialize_new_state(self):
    """ Initializes a new state based on config """
    pass

  def get_max_possible_actions(self):
    """ Get the max amount of possible actions for this game type """
    pass
    
  def get_best_move(self, state):
    """ Get the best move in a given state """
    pass
  
  def get_move_count(self, state):
    """ Get how many moves there are in a given state """
    pass

  def get_all_tree_moves(self, state):  
    """Get every move from a given state in the representation the monte carlo tree should use """
    pass
  
  def execute_move(self, state, move):
    """ Executes a given move in a given state """
    pass
  
  def get_winner_data(self, state):
    """ Get data that corresponds to who is the winner """
    pass

  def hash(self, state):
    """ Return a hash of state for use with TreeState, MUST BE UNIQUE/NO COLLISIONS """
    pass

  def get_state(self, state):
    """ Returns a state representation that will be used by the NN bridge for evaluation """
    pass
  
  def get_all_nn_moves(self, state):
    """ Get all moves in the representation the nn should use """
    pass 

class HexGameBridge(GameBridge):
  def __init__(self, config):
    super().__init__(config)
    
  def initialize_new_state(self):
    return HexGameBoard(self.config.board_type, self.config.size)
  
  def get_state(self, state):
    return state.board

  def get_max_possible_actions(self):
    return self.config.size * self.config.size

  def get_winner_data(self, state):
    winner = state.get_win()
    if winner == Player.PLAYER1:
      to_return = 1
    elif winner == Player.PLAYER2:
      to_return = 0
    else:
      # should not happen but for debugging
      raise Exception("No more moves but no winner yet, critical game logic failure") 
    return to_return
  
  def get_move_count(self, state):
    return len(state.get_all_moves()[1])
  
  def get_all_tree_moves(self, state):
    return state.get_all_moves()[1]
  
  def execute_move(self, state, move):
    new_state = self.initialize_new_state()
    board = state.board
    new_board = [[None for y in range(self.config.size)] for x in range(self.config.size)]
    for row in range(self.config.size):
      for col in range(self.config.size):
        board_element = board[row][col]
        if board_element.state == PegState.PLAYER1:
          new_board[row][col] = Peg((row, col), PegState.PLAYER1)
        elif board_element.state == PegState.PLAYER2:
          new_board[row][col] = Peg((row, col), PegState.PLAYER2)
        else:
          new_board[row][col] = Peg((row, col), PegState.EMPTY)
    new_state.set_board(new_board)
    new_state.do_action(move)
    return new_state


  def get_win(self, state):
    return state.get_win()

  def hash(self, state):
    """
    :param state: TreeNode state
    """
    """if isinstance(state, HexGameBoard):
      board = state.board
    elif isinstance(state, TreeNode):
      board = state.state.board
    else: 
      board = state"""
    # If we assume state is always a state instance (i.e. hexgameboard for hexgameboard)
    board = state.board

    hash_val = ""
    hash_elements = []
    for row in board:
      for col in row:
        hash_elements.append(str(col.state.value))
    hash_val = hash_val.join(hash_elements)
    return hash_val

  def get_all_nn_moves(self, state):
    return state.get_all_moves()[0]

class TreeNode:
  def __init__(self, config, state, player_to_move, tree_state, default_policy, epsilon, game_bridge):
    self.config = config
    self.visit_count = 0
    self.children = {}
    self.parents = []
    self.state = state  # Assume that GameBridge knows how to interpret state
    self.player_to_move = player_to_move
    self.tree_state = tree_state
    self.edge_visit_counts = {}
    self.evaluation_value = 0
    self.default_policy = default_policy
    self.epsilon = epsilon
    self.move_to_child = {}
    self.game_bridge = game_bridge
    self.hash = self.game_bridge.hash(self.state)
    # Function that adds a (root, D) pair to RBUF

  def monte_carlo_action(self):
    for num in range(self.config.rollouts_per_move):
      self.rollout()
    
    distribution = [0 for x in range(self.game_bridge.get_max_possible_actions())]
    child_dist_map = [None for x in range(self.game_bridge.get_max_possible_actions())]

    for child_str in self.children:
      child = self.children[child_str]
      move = self.move_to_child[child.hash]
      edge_visit_count = child.edge_visit_counts[self.hash]
      distribution[move[0][0] * self.config.size + move[0][1]] += edge_visit_count
      child_dist_map[move[0][0] * self.config.size + move[0][1]] = child

    whole_sum = sum(distribution)
    distribution = np.asarray(distribution)/whole_sum
    RBUF_pair = ((self.state, self.player_to_move), distribution)

    best_move = list(distribution).index(max(distribution))
    next_root = child_dist_map[best_move]

    return RBUF_pair, next_root

  def add_parent(self, new_parent):
    self.parents.append(new_parent)
    self.edge_visit_counts[new_parent.hash] = 0

  def add_child(self, child, move):
    self.children[child.hash] = child
    
    # This may be useful for generating D distribution depending on the game
    self.move_to_child[child.hash] = move 
    
  def increment_edge_visit_count(self, parent):
    self.edge_visit_counts[parent.hash] += 1

  def rollout(self, force_rollout = False):
    # If there are no more possible children/ungenerated children
    if self.game_bridge.get_move_count(self.state) == 0:
      # IF there are no more moves there exists a winner
      return self.game_bridge.get_winner_data(self.state)
      
    # If we've started rolling out, continue rolling out
    # If you are the root note, but do not have any children, then expand and generate 
    # your children (i.e. no parents, no children)
    # But if you are not a root node ( you have a parent) and do not have children then you should go over to Rollout
    if (len(self.children) != 0 or len(self.parents) == 0) and not force_rollout:
      if len(self.children) != self.game_bridge.get_move_count(self.state):
        # Generate nodes
        moves = self.game_bridge.get_all_tree_moves (self.state)
        for move in moves: 
          new_state = self.game_bridge.execute_move(self.state, (move, self.player_to_move))
          # If new child node does not exist in tree
          if not self.tree_state.exists(new_state):
            # We assume it is always two player sequential
            player_to_move = Player.PLAYER1
            if self.player_to_move == Player.PLAYER1:
              player_to_move = Player.PLAYER2
            new_monte_carlo_node = TreeNode(self.config, new_state, player_to_move, self.tree_state, self.default_policy, self.epsilon, self.game_bridge)
            new_monte_carlo_node.add_parent(self)
            self.add_child(new_monte_carlo_node, (move, self.player_to_move))
            self.tree_state.add_node(new_monte_carlo_node)
          # If new child node exists in tree but is not added as a child of this node
          elif self.state_hash(new_state) not in self.children.keys():
            self.add_child(self.tree_state.get_node(new_state), (move, self.player_to_move))
            self.children[self.state_hash(new_state)].add_parent(self)
      
      # Select next node using tree policy
      best_child = None
      best_tree_policy_value = 0
      for child_hash in self.children:
        child = self.children[child_hash]
        if best_child == None:
          best_child = child
          
          # From slides
          if child.player_to_move == Player.PLAYER1:
            best_tree_policy_value = child.get_q_value(self) + child.get_u_value(self)
          else:
            best_tree_policy_value = child.get_q_value(self) - child.get_u_value(self)
        else:
          # From slides
          if self.player_to_move == Player.PLAYER1:
            tree_policy_value = child.get_q_value(self) + child.get_u_value(self)
          else:
            tree_policy_value = child.get_q_value(self) - child.get_u_value(self) 
            
            
          if self.player_to_move == Player.PLAYER1 and tree_policy_value > best_tree_policy_value:
            best_tree_policy_value = tree_policy_value
            best_child = child

          elif self.player_to_move == Player.PLAYER2 and tree_policy_value < best_tree_policy_value:
            best_tree_policy_value = tree_policy_value
            best_child = child
      
      # Move to next child
      best_child.increment_edge_visit_count(self)
      evaluation = best_child.rollout()
      self.evaluation_value += evaluation
      self.visit_count += 1
      return evaluation

    else: 
      # rollout policy
      
      # e-greedy
      if random.random() >= self.epsilon:
        # policy
        moves = self.game_bridge.get_all_nn_moves(self.state)
        action = self.default_policy.eval((moves, self.player_to_move, self.game_bridge.get_state(self.state), "greedy"))
      else:
        # random
        moves = self.game_bridge.get_all_tree_moves(self.state)
        action = random.choice(moves)
        action = (action, self.player_to_move)

      new_state = self.game_bridge.execute_move(self.state, action)
      
      # If new child node does not exist in tree
      if not self.tree_state.exists(new_state):
        player_to_move = Player.PLAYER1
        if self.player_to_move == Player.PLAYER1:
          player_to_move = Player.PLAYER2
        new_monte_carlo_node = TreeNode(self.config, new_state, player_to_move, self.tree_state, self.default_policy, self.epsilon, self.game_bridge)
        new_monte_carlo_node.add_parent(self)
        self.add_child(new_monte_carlo_node, action)
        self.tree_state.add_node(new_monte_carlo_node)
      # If new child node exists in tree but is not added as a child of this node
      elif self.state_hash(new_state) not in self.children.keys():
        self.add_child(self.tree_state.get_node(new_state), action)
        self.children[self.state_hash(new_state)].add_parent(self)

      child = self.children[self.state_hash(new_state)]
      child.increment_edge_visit_count(self)
      evaluation = child.rollout(force_rollout = True)
      self.evaluation_value += evaluation
      self.visit_count += 1
      return evaluation
  
  def get_q_value(self, parent):
    if self.edge_visit_counts[parent.hash] == 0:
      return 0.5 # We assume that if a move has never been encountered, it will on average lead to 50% wins, and 50% losses.
    return self.evaluation_value/self.edge_visit_counts[parent.hash]
  
  def get_u_value(self, parent):
    parent_hash = parent.hash
    # self.visit_count must be > 0
    if self.visit_count == 0:
      return self.config.exploration_constant  # Derivative when visit count is 0
    elif self.edge_visit_counts[parent_hash] == 0:
      return self.config.exploration_constant * math.sqrt(math.log(self.visit_count))
    else:
      return self.config.exploration_constant * math.sqrt(math.log(self.visit_count) / (1 + self.edge_visit_counts[parent_hash]))

  def state_hash(self, state):
    return self.game_bridge.hash(state)

  def reset_parents(self):
    self.parents = []
    self.edge_visit_counts = {}
  