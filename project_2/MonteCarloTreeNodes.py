from Pieces import PegState
from HexGameBoard import HexGameBoard
from PlayerEnum import Player
import copy
import math
import numpy as np
import random

class TreeState:
  def __init__(self,):
    self.tree_states = {}
    
  def exists(self, state):
    return self.state_hash(state) in self.tree_states
  
  def get_existing_parents(self, state):
    hashed_state = self.state_hash(state)
    if self.state_hash(state) in self.tree_states:
      return self.tree_states[hashed_state].parents
    else:
      return []

  def state_hash(self, state):
    if isinstance(state, HexGameBoard):
      board = state.board
    elif isinstance(state, TreeNode):
      board = state.hex_board.board
    else: 
      board = state
    # TODO make generic
    hash_val = ""
    for row in board:
      for col in row:
        if col.state == PegState.PLAYER1:
          hash_val += "1"
        elif col.state == PegState.PLAYER2:
          hash_val += "2"
        else: 
          hash_val += "0"
    return hash_val
  
  def add_node(self, node):
    self.tree_states[self.state_hash(node)] = node
  
  def get_node(self, state):
    return self.tree_states[self.state_hash(state)]


# TODO: REFACTOR TO MAKE GENERAL
class TreeNode:
  def __init__(self, config, hex_board, player_to_move, tree_state, default_policy, epsilon):
    self.config = config
    self.visit_count = 0
    self.children = {}
    self.parents = [] 
    self.hex_board = hex_board
    self.player_to_move = player_to_move
    self.tree_state = tree_state
    self.edge_visit_counts = {}
    self.evaluation_value = 0
    self.default_policy = default_policy
    self.epsilon = epsilon
    self.move_to_child = {}
    # Function that adds a (root, D) pair to RBUF

  def monte_carlo_action(self):
    for num in range(self.config.rollouts_per_move):
      self.rollout()
    
    # REFACTOR:
    distribution = [0 for x in range(self.config.size * self.config.size)]
    child_dist_map = [None for x in range(self.config.size * self.config.size)]

    for child_str in self.children:
      child = self.children[child_str]
      move = self.move_to_child[self.state_hash(child.hex_board)]
      edge_visit_count = child.edge_visit_counts[self.state_hash(self.hex_board)]
      distribution[move[0][0] * self.config.size + move[0][1]] += edge_visit_count
      child_dist_map[move[0][0] * self.config.size + move[0][1]] = child
    
    norm = np.linalg.norm(distribution)
    distribution = np.asarray(distribution)/norm
    RBUF_pair = ((self.hex_board, self.player_to_move), distribution)
    
    best_move = list(distribution).index(max(distribution))
    next_root = child_dist_map[best_move]

    return RBUF_pair, next_root

  def add_parent(self, new_parent):
    self.parents.append(new_parent)
    self.edge_visit_counts[self.state_hash(new_parent)] = 0

  def add_child(self, child, move):
    self.children[self.state_hash(child)] = child
    
    # This may be useful for generating D distribution depending on the game
    self.move_to_child[self.state_hash(child)] = move 
    
  def increment_edge_visit_count(self, parent):
    self.edge_visit_counts[self.state_hash(parent)] += 1

  def rollout(self, force_rollout = False):
    # If there are more possible children/ungenerated children
    if len(self.hex_board.get_all_moves()[1]) == 0:
      winner = self.hex_board.get_win()
      if winner == Player.PLAYER1:
        to_return = 1
      elif winner == Player.PLAYER2:
        to_return = 0
      else:
        # should not happen but for debugging
        raise Exception("No more moves but no winner yet, critical game logic failure") 
      return to_return
      
    # If we've started rolling out, continue rolling out
    # If you are the root note, but do not have any children, then expand and generate 
    # your children (i.e. no parents, no children)
    # But if you are not a root node ( you have a parent) and do not have children then you should go over to Rollout
    if (len(self.children) != 0 or len(self.parents) == 0) and not force_rollout:
      if len(self.children) != len(self.hex_board.get_all_moves()[1]):
        # Generate nodes
        moves = self.hex_board.get_all_moves()[1]
        for move in moves: 
          new_hex_board = HexGameBoard(self.config.board_type, self.config.size)
          new_hex_board.set_board(copy.deepcopy(self.hex_board.board))
          new_hex_board.do_action((move, self.player_to_move))
          # If new child node does not exist in tree
          if not self.tree_state.exists(new_hex_board):
            player_to_move = Player.PLAYER1
            if self.player_to_move == Player.PLAYER1:
              player_to_move = Player.PLAYER2
            new_monte_carlo_node = TreeNode(self.config, new_hex_board, player_to_move, self.tree_state, self.default_policy, self.epsilon)
            new_monte_carlo_node.add_parent(self)
            self.add_child(new_monte_carlo_node, (move, self.player_to_move))
            self.tree_state.add_node(new_monte_carlo_node)
          # If new child node exists in tree but is not added as a child of this node
          elif self.state_hash(new_hex_board) not in self.children.keys():
            self.add_child(self.tree_state.get_node(new_hex_board), (move, self.player_to_move))
            self.children[self.state_hash(new_hex_board)].add_parent(self)
      
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
          if child.player_to_move == Player.PLAYER1:
            tree_policy_value = child.get_q_value(self) + child.get_u_value(self)
          else:
            tree_policy_value = child.get_q_value(self) - child.get_u_value(self) 
            
            
          if tree_policy_value > best_tree_policy_value:
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
      moves = self.hex_board.get_all_moves()

      # e-greedy
      if random.random() >= self.epsilon: 
        action = self.default_policy.eval((moves[0], self.player_to_move, self.hex_board.get_state(), self))
      else:
        action = (random.choice(moves[1]), self.player_to_move)

      new_hex_board = HexGameBoard(self.config.board_type, self.config.size)
      new_hex_board.set_board(copy.deepcopy(self.hex_board.board))
      new_hex_board.do_action(action)
      # If new child node does not exist in tree
      if not self.tree_state.exists(new_hex_board):
        player_to_move = Player.PLAYER1
        if self.player_to_move == Player.PLAYER1:
          player_to_move = Player.PLAYER2
        new_monte_carlo_node = TreeNode(self.config, new_hex_board, player_to_move, self.tree_state, self.default_policy, self.epsilon)
        new_monte_carlo_node.add_parent(self)
        self.add_child(new_monte_carlo_node, action)
        self.tree_state.add_node(new_monte_carlo_node)
      # If new child node exists in tree but is not added as a child of this node
      elif self.state_hash(new_hex_board) not in self.children.keys():
        self.add_child(self.tree_state.get_node(new_hex_board), action)
        self.children[self.state_hash(new_hex_board)].add_parent(self)

      child = self.children[self.state_hash(new_hex_board)]
      child.increment_edge_visit_count(self)
      evaluation = child.rollout(force_rollout = True)
      self.evaluation_value += evaluation
      self.visit_count += 1
      return evaluation
  
  def get_q_value(self, parent):
    if self.edge_visit_counts[self.state_hash(parent)] == 0:
      return 0
    return self.evaluation_value/self.edge_visit_counts[self.state_hash(parent)]
  
  def get_u_value(self, parent):
    parent_hash = self.state_hash(parent)
    # self.visit_count must be > 0
    if self.visit_count == 0:
      return 0
    elif self.edge_visit_counts[parent_hash] == 0:
      return self.config.exploration_constant * math.sqrt(math.log(self.visit_count))
    else:
      return self.config.exploration_constant * math.sqrt(math.log(self.visit_count) / (1 + self.edge_visit_counts[parent_hash]))

  def state_hash(self, state):
    # POSSIBLE OPTIMIZATION: CALCULATE STATE HASH FOR EACH NODE ONCE AND ONLY ONCE
    if isinstance(state, HexGameBoard):
      board = state.board
    elif isinstance(state, TreeNode):
      board = state.hex_board.board
    else: 
      board = state
      
    # TODO make generic
    hash_val = ""
    for row in board:
      for col in row:
        if col.state == PegState.PLAYER1:
          hash_val += "1"
        elif col.state == PegState.PLAYER2:
          hash_val += "2"
        else: 
          hash_val += "0"
    return hash_val

  def reset_parents(self):
    self.parents = []
    self.edge_visit_counts = {}

