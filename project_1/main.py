from ConfigReader import ConfigReader
from SimWorldDisplayer import ImageDisplay
from pegSolitaireBoard import PegSolitaireBoard
from simWorld import ShapeType
from ReinforcementLearner import ReinforcementLearner
from PegSolitairePlayer import PegSolitairePlayer
<<<<<<< Updated upstream
import random
import json
from tqdm import tqdm
hyperparams = []
=======
from tqdm import tqdm
>>>>>>> Stashed changes

def param_brute():
  for num in tqdm(range(0, 100), "Possible params checked"):
    base_loss = random.uniform(0, 1)
    peg_loss_mod = random.uniform(0, 1)
    win = random.choice([1, 10, 100, 1000, 10000])
    critic_discount = random.uniform(0.5, 1)
    actor_discount = random.uniform(0.5, 1)
    critic_eligibility = random.uniform(0.5, 1)
    actor_eligibility = random.uniform(0.5, 1)
    initial_epsilon = random.uniform(0.25, 1)

    config = ConfigReader()
    config.actor_eligibility_decay_rate = actor_eligibility
    config.critic_eligibility_decay_rate = critic_eligibility
    config.actor_discount_factor = actor_discount
    config.critic_discount_factor = critic_discount
    config.initial_epsilon = initial_epsilon
    config.positive_reward_on_win = win
    config.negative_reward_on_loss_base = base_loss
    config.negative_reward_on_loss_per_peg = peg_loss_mod
    best_score = run_tests(5, config)

    best_score = sum(best_score)/len(best_score)
    hyperparams.append([str((base_loss, peg_loss_mod, win, critic_discount, actor_discount, critic_eligibility, actor_eligibility, initial_epsilon)), best_score])
    #print('Number of correct runs:', sum([1 for ele in learner.peg_log if ele == 1]))
    #learner.display_game()

  try:
    with open('file.txt', 'a') as file:
      hyperparams.sort(key=lambda x: x[1], reverse=True)
      file.write(json.dumps(hyperparams))
  except:
    pass
  finally:
    print(hyperparams)

def iter_round(config):
  config_list = [
    config.actor_eligibility_decay_rate,
    config.critic_eligibility_decay_rate,
    config.actor_discount_factor,
    config.critic_discount_factor,
    config.initial_epsilon,
    config.negative_reward_on_loss_base,
    config.negative_reward_on_loss_per_peg
  ]
  attribute_index = random.choice(range(0, len(config_list)))
  attribute = config_list[attribute_index]
  old_val = attribute + 1  # copy value
  old_val -= 1
  candidate_changes = [random.uniform(-0.1, 0.1) for x in range(0, 1)]
  candidates = [attribute+candidate_changes[i] for i in range(0, 1)]
  scores = []
  for cand in candidates:
    config_list[attribute_index] = cand
    load_config(config, config_list)
    avg_score = sum(run_tests(5, config))/5
    scores.append([avg_score, cand])
  scores.sort(key=lambda x: x[0])
  config_list[attribute_index] = scores[0][1]
  load_config(config, config_list)
  return scores[0][0]

def load_config(config, vals):
  config.actor_eligibility_decay_rate = vals[0]
  config.critic_eligibility_decay_rate = vals[1]
  config.actor_discount_factor = vals[2]
  config.critic_discount_factor = vals[3]
  config.initial_epsilon = vals[4]
  config.negative_reward_on_loss_base = vals[5]
  config.negative_reward_on_loss_per_peg = vals[6]

def iter_search(rounds, params):
  base_loss = params[5]
  peg_loss_mod = params[6]
  win = 1000
  critic_discount =params[3]
  actor_discount =  params[2]
  critic_eligibility = params[1]
  actor_eligibility = params[0]
  initial_epsilon = params[4]

  config = ConfigReader()
  config.actor_eligibility_decay_rate = actor_eligibility
  config.critic_eligibility_decay_rate = critic_eligibility
  config.actor_discount_factor = actor_discount
  config.critic_discount_factor = critic_discount
  config.initial_epsilon = initial_epsilon
  config.positive_reward_on_win = win
  config.negative_reward_on_loss_base = base_loss
  config.negative_reward_on_loss_per_peg = peg_loss_mod
  results = []
  for r in tqdm(range(0, rounds), "progress"):
    result = iter_round(config)
    result_str = str(config.actor_eligibility_decay_rate) + ", " + \
                   str(config.critic_eligibility_decay_rate) + ", " + \
                   str(config.actor_discount_factor) + ", " + \
                   str(config.critic_discount_factor) + ", " + \
                   str(config.initial_epsilon) + ", " + \
                   str(config.positive_reward_on_win) + ", " + \
                   str(config.negative_reward_on_loss_base) + ", " + \
                   str(config.negative_reward_on_loss_per_peg) + " | " + \
                    str(result) + "\n"
    results.append(result_str)
  with open('file2.txt', 'a') as file:
    for res in results:
      file.write(res)







def run_tests(count, config):
  best_score = []
  for num2 in range(0, count):
    player = PegSolitairePlayer(config.empty_cells, config.board_type, config.size, config.image_size,
                                config.frame_delay, config.positive_reward_on_win, config.negative_reward_on_loss_base, config.negative_reward_on_loss_per_peg)
    learner = ReinforcementLearner(player, config)

    learner.fit()
    corrects = sum([1 for ele in learner.peg_log if ele == 1])
    best_score.append(corrects)
  return best_score

#p = (0.8972402459603667, 0.8931089114734135, 0.9409684975677948, 0.602227525679401, 0.30862613139410333, 0.27798283333040524, 0.4762362992036559)
#iter_search(10, p)
config = ConfigReader()

<<<<<<< Updated upstream
player = PegSolitairePlayer(config.empty_cells, config.board_type, config.size, config.image_size, config.frame_delay, config.positive_reward_on_win, config.negative_reward_on_loss_base, config.negative_reward_on_loss_per_peg)
learner = ReinforcementLearner(player, config)

learner.fit()
print('Number of correct runs:', sum([1 for ele in learner.peg_log if ele == 1]))
learner.display_game()
=======
#player = PegSolitairePlayer(config.empty_cells, config.board_type, config.size, config.image_size, config.frame_delay, config.win, config.base_loss, config.peg_loss)
#learner = ReinforcementLearner(player, config)
#
#learner.fit()
#print('Number of correct runs:', sum([1 for ele in learner.peg_log if ele == 1]))
#learner.display_game()

# FOR TESTING:
res = []
for num in tqdm(range(0, 20), desc="Progress"):
  player = PegSolitairePlayer(config.empty_cells, config.board_type, config.size, config.image_size, config.frame_delay, config.win, config.base_loss, config.peg_loss)
  learner = ReinforcementLearner(player, config)

  learner.fit()
  res.append(sum([1 for ele in learner.peg_log if ele == 1]))

print(res)
avg = sum(res)/len(res)
print(avg)
res2 = [abs(x - avg)**2 for x in res]
std = sum(res2)/len(res)
print(std)



'''
# board = PegSolitaireBoard(config.empty_cells, config.board_type, config.size)
img_display = ImageDisplay(board)
img_display.display()

moves = board.get_all_moves()


print(moves)

board.do_action(moves[0])

img_display.display()

# img_display.display()
'''