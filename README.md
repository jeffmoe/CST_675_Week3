# CST_675_Week3
Implement and Compare Q-Learning and REINFORCE in Python

## Goal
Compare a **value-based** method (tabular Q-Learning) with a **policy-based** method
(REINFORCE) on the CartPole-v1 environment from OpenAI Gym / Gymnasium.

## Setup
Requires Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt # gymnasium, numpy, matplotlib, torch
python cartpole_rl.py
```

The script uses **Gymnasium** (`import gymnasium as gym`), the maintained successor
to OpenAI Gym, with the 5-value `step()` API:
`obs, reward, terminated, truncated, info`. All RNGs (Python, NumPy, PyTorch, and the
environment) are seeded with `SEED = 42`, so runs are reproducible.

**CartPole-v1 at a glance:** 4-D continuous observation (cart position, cart velocity,
pole angle, pole angular velocity), 2 discrete actions (push left/right), +1 reward per
step, episodes capped at 500 steps. The two velocity components are unbounded, so
Q-Learning needs clipped ranges for them when the state is discretized.

## Deliverables
- [x] Working Python implementations of both agents (tabular Q-Learning + REINFORCE)
- [ ] Training metrics and plots
- [ ] Strengths/weaknesses analysis
- [ ] 6–7 slide PowerPoint with speaker notes and APA references

## Tabular Q-Learning (`cartpole_rl.py`, section 2)

**State discretization.** `discretize_state()` clips each observation dimension to a
range in `STATE_BOUNDS` and maps it to one of `bins = (6, 6, 12, 12)` equal-width bins:

| Dimension | Range | Bins |
|---|---|---|
| Cart position | ±4.8 (env bound) | 6 |
| Cart velocity | ±3.0 (manual; env bound is ±inf) | 6 |
| Pole angle | ±0.418 rad (env bound) | 12 |
| Pole angular velocity | ±3.5 (manual; env bound is ±inf) | 12 |

**Q-table.** Shape `bins + (n_actions,)` = `(6, 6, 12, 12, 2)`; `q_table[s]` holds the
value of each action in discrete state `s`.

**Exploration.** ε-greedy with per-episode decay: ε starts at 1.0, is multiplied by 0.995
after each episode, and is floored at 0.01 (reached around episode 920).

**Update.** Bellman/TD update `Q(s,a) += α [target − Q(s,a)]` with α = 0.1, γ = 0.99:
- **Terminated** (pole fell / cart out of bounds): `target = r` — no future value.
- **Truncated** (500-step time limit): `target = r + γ · max_a' Q(s', a')` — the state
  isn't truly terminal, so the agent still bootstraps.

`q_learning(env, ...)` returns `(rewards, epsilons, q_table)`.

**Changes from the starter code:**
1. Manual clipping ranges for the two velocities (the original `linspace` over ±inf
   bounds produced unusable bins).
2. Decaying ε instead of a fixed ε = 0.1.
3. No bootstrapping on termination; bootstrapping kept on truncation.

**Current result** (seed 42, 1,000 episodes, ~11 s): mean reward over the last 100
episodes = **71.1**, best episode = 212.

## REINFORCE (`cartpole_rl.py`, section 3)

**Policy network.** `PolicyNetwork` takes the raw 4-D state (no discretization) through
`Linear(4, 128) → ReLU → Linear(128, 2) → softmax` to get π(a | s; θ).

**Exploration.** Actions are sampled from π via `Categorical`, so exploration comes from
the policy's own stochasticity — no ε schedule.

**Update.** Once per episode (Monte Carlo):
1. Compute discounted returns `G_t = r_t + γ G_{t+1}` (γ = 0.99) with `discounted_returns()`.
2. Normalize returns to zero mean / unit std — a simple baseline that reduces variance.
3. Loss `= −Σ_t log π(a_t | s_t) · G_t`, minimized with Adam (lr = 1e-2).

No bootstrapping, so truncation at 500 steps needs no special case.

`reinforce(env, ...)` returns `(rewards, losses, policy)`. It trains on a fresh seeded
environment (`make_env()`) with `torch.manual_seed(SEED)` reset, so both agents start
from the same conditions.

**Current result** (seed 42, 1,000 episodes, ~65 s): mean reward over the last 100
episodes = **333.4**, best episode = 500 (the cap).

## Background: Value-Based vs. Policy-Based RL

| | Value-based (Q-Learning) | Policy-based (REINFORCE) |
|---|---|---|
| What it learns | Action values Q(s, a) | Policy π(a \| s; θ) directly |
| How it acts | Greedy / ε-greedy over Q | Samples from π |
| Update signal | Bootstrapped TD error, every step | Monte Carlo return, end of episode |
| Data use | Off-policy, can reuse experience | On-policy, discards data after each update |
| Action spaces | Discrete only (needs argmax) | Discrete or continuous |
| Variance / bias | Lower variance, biased by bootstrapping | Unbiased gradient, high variance |
| CartPole specifics | Continuous state must be discretized into bins | Uses raw state with a small neural network |

**In one line:** Q-Learning asks "how good is each action?" and acts greedily on the
answer; REINFORCE asks "which way should I nudge the action probabilities?" and
follows the gradient of expected return.
