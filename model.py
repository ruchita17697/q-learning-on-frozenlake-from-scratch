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

# Step 5 - should_explore
def should_explore(epsilon, rng):
   return bool(epsilon> rng.random())
pass

# Step 6 - epsilon_greedy_action
import numpy as np


def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)
    return greedy_action(q_table, state)

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

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # TODO: reset env, then repeatedly call interaction_step until done or max_steps, returning total reward.
    state, _ = env.reset()

    total_reward = 0.0

    for _ in range(max_steps):
        next_state, reward, done = interaction_step(
            env,
            q_table,
            state,
            epsilon,
            alpha,
            gamma,
            rng,
        )

        total_reward += reward
        state = next_state

        if done:
            break

    return float(total_reward)
    pass

# Step 13 - train_q_learning
def train_q_learning(
    env,
    num_episodes,
    alpha=0.8,          # ← high alpha learns faster on stochastic envs
    gamma=0.95,         # ← slightly lower gamma
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.99,
    seed=0,
    max_steps=200,
):
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)

    q_table = init_q_table(env.observation_space.n, env.action_space.n)
    episode_returns = []
    epsilon = epsilon_start

    for _ in range(num_episodes):
        episode_return = run_training_episode(
            env, q_table, epsilon, alpha, gamma, rng, max_steps
        )
        episode_returns.append(float(episode_return))
        epsilon = decay_epsilon(epsilon, epsilon_decay, epsilon_min)

    return q_table, episode_returns

# Step 14 - extract_greedy_policy
def extract_greedy_policy(q_table):
    # TODO: return a 1D int64 array mapping each state to its best (argmax) action.\
     return np.argmax(q_table, axis=1).astype(np.int64)

# Step 15 - run_greedy_episode
def run_greedy_episode(env, policy, seed=None, max_steps=200):
    """Run one greedy episode and return True if the goal was reached."""
    # TODO: reset env, follow policy[state] each step, return bool(success)
    
    reset_kwargs = {"seed": seed} if seed is not None else {}
    state, _ = env.reset(**reset_kwargs)

    for _ in range(max_steps):
        action = int(policy[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            return reward > 0

        state = next_state

    return False

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

