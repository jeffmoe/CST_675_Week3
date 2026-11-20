"""
Q-Learning vs. REINFORCE on CartPole-v1.

Install dependencies (from the project root, inside your virtual environment):
    pip install gymnasium numpy matplotlib torch
or:
    pip install -r requirements.txt
"""

# =============================================================================
# 1. Setup
# =============================================================================
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

# --- Reproducibility ---------------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Environment --------------------------------------------------------------
ENV_ID = "CartPole-v1"


def make_env(seed: int = SEED) -> gym.Env:
    """Create a seeded CartPole-v1 environment.

    Gymnasium seeds the environment's RNG on the first reset(seed=...).
    The action space has its own RNG (used by action_space.sample()),
    so it is seeded separately.
    """
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


env = make_env()

print(f"Environment:       {ENV_ID}")
print(f"Observation space: {env.observation_space}")
print(f"  low:  {env.observation_space.low}")
print(f"  high: {env.observation_space.high}")
print(f"Action space:      {env.action_space}  (0 = push left, 1 = push right)")
print(f"Max episode steps: {env.spec.max_episode_steps}")

# Sanity check of the 5-value step API: obs, reward, terminated, truncated, info
obs, info = env.reset(seed=SEED)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(f"Sample step -> obs={obs}, reward={reward}, "
      f"terminated={terminated}, truncated={truncated}")


# =============================================================================
# 2. Tabular Q-Learning (value-based)
# =============================================================================
# Clipping ranges per state dimension: (cart pos, cart vel, pole angle, pole ang vel).
# observation_space reports ±inf for the two velocities, which would make
# linspace bins useless, so those use manual ranges. Position/angle use env bounds.
STATE_BOUNDS = [
    (env.observation_space.low[0], env.observation_space.high[0]),  # ±4.8 m
    (-3.0, 3.0),                                                    # cart velocity
    (env.observation_space.low[2], env.observation_space.high[2]),  # ±0.418 rad
    (-3.5, 3.5),                                                    # pole angular velocity
]


def discretize_state(state, bins=(6, 6, 12, 12), bounds=STATE_BOUNDS):
    """Map a continuous observation to a tuple of bin indices (one per dimension)."""
    discretized = []
    for i, (low, high) in enumerate(bounds):
        val = np.clip(state[i], low, high)
        edges = np.linspace(low, high, bins[i] + 1)
        idx = np.digitize(val, edges) - 1           # 0-based bin index
        discretized.append(int(np.clip(idx, 0, bins[i] - 1)))  # val == high -> last bin
    return tuple(discretized)


def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
               bins=(6, 6, 12, 12)):
    """Train a tabular Q-Learning agent. Returns (episode rewards, epsilons, Q-table)."""
    # Q-table shape: bins + (n_actions,) -> (6, 6, 12, 12, 2).
    # q_table[s] is the vector of action values for discrete state s.
    q_table = np.zeros(bins + (env.action_space.n,))
    rewards_q_learning, epsilons = [], []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(state, bins)
        done = False
        total_reward = 0.0

        while not done:
            # Epsilon-greedy: explore with prob. epsilon, otherwise exploit best known action.
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize_state(next_obs, bins)

            # Bellman (TD) update: Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)].
            # Terminated (pole fell / cart out) -> no future value, so target = r.
            # Truncated (500-step time limit) -> state isn't terminal, so still bootstrap.
            if terminated:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(q_table[next_state])
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
            total_reward += reward

        rewards_q_learning.append(total_reward)
        epsilons.append(epsilon)
        # Decay exploration once per episode, never below epsilon_min.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return rewards_q_learning, epsilons, q_table


rewards_q_learning, epsilons_q_learning, q_table = q_learning(env)

last100 = np.mean(rewards_q_learning[-100:])
print(f"\nQ-Learning: mean reward (last 100 episodes) = {last100:.1f}, "
      f"best = {max(rewards_q_learning):.0f}, final epsilon = {epsilons_q_learning[-1]:.3f}")
