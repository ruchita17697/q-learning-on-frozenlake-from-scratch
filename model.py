"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    
    return np.zeros((num_states, num_actions), dtype=np.float64)
    pass

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    return np.max(q_table[state])
    pass

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
     return int (np.argmax (q_table[state]))
pass

# Step 4 - sample_random_action
def sample_random_action(action_space):
     return int(action_space.sample())
pass

# Step 5 - should_explore
def should_explore(epsilon, rng):
   return bool(epsilon> rng.random())
pass

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    if rng.random() < epsilon:
        return sample_random_action(action_space)
    else:
        return greedy_action(q_table, state)
    pass

# Step 7 - decay_epsilon (not yet solved)
# TODO: implement

# Step 8 - td_target (not yet solved)
# TODO: implement

# Step 9 - td_error (not yet solved)
# TODO: implement

# Step 10 - q_learning_update (not yet solved)
# TODO: implement

# Step 11 - interaction_step (not yet solved)
# TODO: implement

# Step 12 - run_training_episode (not yet solved)
# TODO: implement

# Step 13 - train_q_learning (not yet solved)
# TODO: implement

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

