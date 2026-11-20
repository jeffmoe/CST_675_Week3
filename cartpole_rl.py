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
# - Velocities: observation_space reports ±inf, so use manual ranges.
# - Position/angle: use the *termination* thresholds (±2.4 m, ±12° ≈ ±0.2095 rad),
#   not observation_space (±4.8, ±0.418), which is 2x wider -> half the bins unreachable.
X_LIMIT = env.unwrapped.x_threshold                 # 2.4
THETA_LIMIT = env.unwrapped.theta_threshold_radians  # 0.2095
STATE_BOUNDS = [
    (-X_LIMIT, X_LIMIT),          # cart position
    (-3.0, 3.0),                  # cart velocity
    (-THETA_LIMIT, THETA_LIMIT),  # pole angle
    (-3.5, 3.5),                  # pole angular velocity
]

# Bins per dimension. Only pole angle and angular velocity are resolved; cart position
# and velocity get 1 bin (ignored). A smaller table (72 states vs. 5,184 for 6x6x12x12)
# gets visited far more often, so it learns within 1,000 episodes. Chosen by a small
# grid search over bounds, bins, alpha, and epsilon decay on seeds 42-44.
QL_BINS = (1, 1, 6, 12)


def discretize_state(state, bins=QL_BINS, bounds=STATE_BOUNDS):
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
               bins=QL_BINS):
    """Train a tabular Q-Learning agent. Returns (episode rewards, epsilons, Q-table)."""
    # Q-table shape: bins + (n_actions,) -> (1, 1, 6, 12, 2).
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


# =============================================================================
# 3. REINFORCE (policy-based, Monte Carlo policy gradient)
# =============================================================================
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """pi(a | s; theta): maps the raw 4-D state to action probabilities (no discretization)."""

    def __init__(self, obs_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def discounted_returns(rewards, gamma):
    """G_t = r_t + gamma * G_{t+1}, computed backwards over one episode."""
    returns, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    return list(reversed(returns))


def reinforce(env, num_episodes=1000, lr=5e-3, gamma=0.99, hidden=128):
    """Train a REINFORCE agent. Returns (episode rewards, losses, policy network)."""
    # lr=5e-3: at 1e-2 the policy collapsed to one action within ~10 episodes on seed 44.
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, hidden)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    rewards_reinforce, losses = [], []

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs, ep_rewards = [], []
        done = False

        # Roll out one full episode with the current (stochastic) policy.
        while not done:
            probs = policy(torch.as_tensor(state, dtype=torch.float32))
            dist = Categorical(probs)          # sampling = built-in exploration (no epsilon)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            ep_rewards.append(reward)

        # Monte Carlo returns; normalizing them acts as a simple baseline to cut variance.
        # (No bootstrapping, so truncation needs no special case: G_t just stops at 500.)
        returns = torch.tensor(discounted_returns(ep_rewards, gamma), dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient: maximize sum_t log pi(a_t|s_t) * G_t  ->  minimize its negative.
        loss = -(torch.stack(log_probs) * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_reinforce.append(sum(ep_rewards))
        losses.append(loss.item())

    return rewards_reinforce, losses, policy


# =============================================================================
# 4. Training & evaluation
# =============================================================================
import time

NUM_EPISODES = 1000
SEEDS = [42, 43, 44]          # 3 seeds per method -> report mean ± std
SOLVED_THRESHOLD = 195        # classic CartPole "solved" bar
MA_WINDOW = 100


def set_global_seeds(seed):
    """Seed Python, NumPy, and PyTorch RNGs (the env is seeded via make_env)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(rewards, window=MA_WINDOW):
    """Trailing mean over `window` episodes; NaN until a full window exists."""
    rewards = np.asarray(rewards, dtype=float)
    ma = np.full_like(rewards, np.nan)
    if len(rewards) >= window:
        ma[window - 1:] = np.convolve(rewards, np.ones(window) / window, mode="valid")
    return ma


def compute_metrics(rewards, train_time):
    """Per-run metrics used in the comparison table."""
    rewards = np.asarray(rewards, dtype=float)
    ma = moving_average(rewards)
    hits = np.flatnonzero(ma >= SOLVED_THRESHOLD)   # NaN comparisons are False
    return {
        "rewards": rewards,
        "moving_avg": ma,
        # 1-based episode at which the 100-episode MA first reaches 195; None if never.
        "convergence_episode": int(hits[0]) + 1 if hits.size else None,
        # Reward >= 195 (this also covers every episode that reaches the 500 cap).
        "successful_episodes": int(np.sum(rewards >= SOLVED_THRESHOLD)),
        "final_ma": float(ma[-1]),
        "train_time_s": train_time,
    }


def run_method(name, seed):
    """Train one method on one seed and return its metrics (plus extras for plotting)."""
    set_global_seeds(seed)
    env_run = make_env(seed)
    start = time.perf_counter()
    if name == "Q-Learning":
        rewards, extra, model = q_learning(env_run, num_episodes=NUM_EPISODES)  # extra = epsilons
    else:
        rewards, extra, model = reinforce(env_run, num_episodes=NUM_EPISODES)   # extra = losses
    elapsed = time.perf_counter() - start
    env_run.close()
    metrics = compute_metrics(rewards, elapsed)
    metrics.update(seed=seed, extra=extra, model=model)
    return metrics


def fmt_mean_std(values, fmt="{:.1f}"):
    """'mean ± std' string; std is the sample std (ddof=1) when n > 1."""
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=1) if values.size > 1 else 0.0
    return f"{fmt.format(values.mean())} ± {fmt.format(std)}"


def summarize(runs):
    """Aggregate one method's runs across seeds into table-ready strings."""
    conv = [r["convergence_episode"] for r in runs if r["convergence_episode"] is not None]
    conv_str = f"{fmt_mean_std(conv, '{:.0f}')} ({len(conv)}/{len(runs)})" if conv \
        else f"never (0/{len(runs)})"
    return {
        "Converged at episode": conv_str,
        "Successful episodes": fmt_mean_std([r["successful_episodes"] for r in runs], "{:.0f}"),
        "Final 100-ep MA": fmt_mean_std([r["final_ma"] for r in runs]),
        "Best episode": fmt_mean_std([r["rewards"].max() for r in runs], "{:.0f}"),
        "Training time (s)": fmt_mean_std([r["train_time_s"] for r in runs]),
    }


METHODS = ["Q-Learning", "REINFORCE"]
results = {m: [] for m in METHODS}

print(f"\nTraining {len(METHODS)} methods x {len(SEEDS)} seeds x {NUM_EPISODES} episodes")
for method in METHODS:
    for seed in SEEDS:
        run = run_method(method, seed)
        results[method].append(run)
        conv = run["convergence_episode"] or "never"
        print(f"  {method:<11} seed={seed}: converged={conv!s:>5}  "
              f"successes={run['successful_episodes']:>4}  "
              f"final MA={run['final_ma']:6.1f}  time={run['train_time_s']:5.1f}s")

# --- Summary comparison table (mean ± std over seeds) ------------------------
summaries = {m: summarize(results[m]) for m in METHODS}
rows = list(summaries[METHODS[0]].keys())
col_w = max(len(v) for s in summaries.values() for v in s.values()) + 2
row_w = max(len(r) for r in rows) + 2

print(f"\nSummary over seeds {SEEDS} (mean ± std; convergence = first episode "
      f"with {MA_WINDOW}-ep MA >= {SOLVED_THRESHOLD})")
header = f"{'Metric':<{row_w}}" + "".join(f"{m:>{col_w}}" for m in METHODS)
print(header)
print("-" * len(header))
for row in rows:
    print(f"{row:<{row_w}}" + "".join(f"{summaries[m][row]:>{col_w}}" for m in METHODS))


# =============================================================================
# 5. Plots (seaborn, presentation-ready PNGs)
# =============================================================================
from pathlib import Path

import pandas as pd
import seaborn as sns

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
FIG_SIZE = (13.33, 7.5)   # 16:9, matches a standard widescreen slide in inches
DPI = 200

# Fixed color per method (same in every figure) + neutral ink for text/reference lines.
COLORS = {"Q-Learning": "#2a78d6", "REINFORCE": "#eb6834"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

sns.set_theme(context="talk", style="whitegrid", rc={
    "axes.edgecolor": GRID, "grid.color": GRID, "axes.labelcolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
    "axes.titleweight": "bold", "axes.titlesize": 20, "axes.titlepad": 12,
    "legend.frameon": False, "savefig.bbox": "tight", "figure.dpi": 100,
})


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def mark_threshold(ax):
    """Dashed 'solved' reference line, labeled at its right end."""
    ax.axhline(SOLVED_THRESHOLD, color=INK_MUTED, lw=1.5, ls="--", zorder=1)
    ax.text(1.0, SOLVED_THRESHOLD, f" solved ({SOLVED_THRESHOLD})", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=13, color=INK_MUTED)


episodes = np.arange(1, NUM_EPISODES + 1)

# --- (1) Reward per episode + 100-episode moving average ----------------------
# Raw = mean reward across seeds per episode (thin, faint); MA = mean of per-seed MAs
# (bold) with a ±1 SD band across seeds.
fig, ax = plt.subplots(figsize=(FIG_SIZE[0], 6.3))  # shorter: legend sits below -> ~16:9 overall
for method in METHODS:
    raw = np.stack([r["rewards"] for r in results[method]])
    ma = np.stack([r["moving_avg"] for r in results[method]])
    ma_mean, ma_std = ma.mean(axis=0), ma.std(axis=0, ddof=1)
    c = COLORS[method]
    ax.plot(episodes, raw.mean(axis=0), color=c, lw=0.8, alpha=0.3,
            label=f"{method}: reward per episode")
    ax.plot(episodes, ma_mean, color=c, lw=3, label=f"{method}: {MA_WINDOW}-episode moving avg")
    ax.fill_between(episodes, ma_mean - ma_std, ma_mean + ma_std, color=c, alpha=0.15, lw=0)
mark_threshold(ax)
ax.set(xlim=(1, NUM_EPISODES), ylim=(0, 520), xlabel="Episode", ylabel="Episode reward (steps balanced)")
ax.set_title("CartPole-v1 learning curves: Q-Learning vs. REINFORCE", loc="left")
ax.text(0.0, -0.2, f"Mean across {len(SEEDS)} seeds {SEEDS}; shaded band = ±1 SD of the moving average. "
        "Episodes are capped at 500 steps.", fontsize=13, color=INK_MUTED, transform=ax.transAxes)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=14)
save(fig, "learning_curves.png")

# --- (1b) Same, one panel per seed (shows run-to-run variance) ----------------
fig, axes = plt.subplots(1, len(SEEDS), figsize=FIG_SIZE, sharey=True)
for ax, i in zip(axes, range(len(SEEDS))):
    for method in METHODS:
        run, c = results[method][i], COLORS[method]
        ax.plot(episodes, run["rewards"], color=c, lw=0.6, alpha=0.3)
        ax.plot(episodes, run["moving_avg"], color=c, lw=2.5, label=method)
    ax.axhline(SOLVED_THRESHOLD, color=INK_MUTED, lw=1.2, ls="--", zorder=1)
    ax.set(title=f"Seed {SEEDS[i]}", xlim=(1, NUM_EPISODES), ylim=(0, 520), xlabel="Episode")
    ax.title.set_fontsize(16)
axes[0].set_ylabel("Episode reward")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=14)
fig.suptitle("Learning curves per seed (faint = per-episode reward, bold = 100-episode moving avg, "
             "dashed = solved at 195)", x=0.01, ha="left", fontsize=15, color=INK_MUTED)
fig.tight_layout()
save(fig, "learning_curves_per_seed.png")

# --- (2) Bar chart: convergence episode, successful episodes, training time ----
# Three panels because the metrics have different units (no shared/dual axis).
# Bar = mean over seeds, error bar = ±1 SD, dots = individual seeds.
rows = []
for method in METHODS:
    for r in results[method]:
        rows.append({"Method": method, "Seed": r["seed"],
                     "Convergence episode": r["convergence_episode"],   # None if never converged
                     "Successful episodes": r["successful_episodes"],
                     "Training time (s)": r["train_time_s"]})
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(FIG_DIR / "metrics_by_seed.csv", index=False)

panels = [("Convergence episode", "lower is better", "{:.0f}"),
          ("Successful episodes", f"reward ≥ {SOLVED_THRESHOLD}, higher is better", "{:.0f}"),
          ("Training time (s)", "wall-clock, lower is better", "{:.1f}")]
fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
for ax, (metric, note, fmt) in zip(axes, panels):
    sns.barplot(data=metrics_df, x="Method", y=metric, hue="Method", palette=COLORS,
                errorbar="sd", capsize=0.15, err_kws={"color": INK, "linewidth": 1.5},
                width=0.6, saturation=1, ax=ax, legend=False)
    sns.stripplot(data=metrics_df, x="Method", y=metric, color="white", edgecolor=INK,
                  linewidth=1.2, size=8, jitter=0.12, ax=ax, zorder=3)
    for bar, method in zip(ax.patches, METHODS):   # mean value above each bar
        vals = metrics_df.loc[metrics_df.Method == method, metric].dropna()
        top = vals.mean() + (vals.std(ddof=1) if len(vals) > 1 else 0)
        ax.annotate(fmt.format(vals.mean()), (bar.get_x() + bar.get_width() / 2, top),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK)
    ax.set_title(metric, fontsize=17)
    ax.set(xlabel="", ylabel="")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.text(0.5, -0.14, note, transform=ax.transAxes, ha="center", fontsize=13, color=INK_MUTED)
fig.suptitle(f"Q-Learning vs. REINFORCE: mean over {len(SEEDS)} seeds "
             "(bar = mean, whisker = ±1 SD, dots = individual seeds)",
             x=0.01, ha="left", fontsize=16, color=INK_MUTED)
fig.tight_layout()
save(fig, "metrics_comparison.png")
