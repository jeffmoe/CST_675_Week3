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
