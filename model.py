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

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
   return max(min_epsilon, epsilon * decay_rate)
   pass

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    if done :
        return float (reward)
    else :
        return float (reward + gamma * max_q_value(q_table, next_state))
    pass

# Step 9 - td_error
def td_error(target, q_table, state, action):
    return float (target-q_table[state,action])
    pass

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    target = td_target(reward, gamma, q_table, next_state, done)

    q_table[state, action] += alpha * td_error(
        target, q_table, state, action
    )

    return float(q_table[state, action])

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    action = epsilon_greedy_action(
        q_table,
        state,
        epsilon,
        env.action_space,
        rng,
    )

    next_state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

    q_learning_update(
        q_table,
        state,
        action,
        reward,
        next_state,
        done,
        alpha,
        gamma,
    )

    return int(next_state), float(reward), bool(done)

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

