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
- [x] Training metrics and plots
- [ ] Strengths/weaknesses analysis
- [ ] 6–7 slide PowerPoint with speaker notes and APA references

## Tabular Q-Learning (`cartpole_rl.py`, section 2)

**State discretization.** `discretize_state()` clips each observation dimension to a
range in `STATE_BOUNDS` and maps it to one of `QL_BINS = (1, 1, 6, 12)` equal-width bins:

| Dimension | Range | Bins |
|---|---|---|
| Cart position | ±2.4 m (termination threshold; env bound ±4.8) | 1 (ignored) |
| Cart velocity | ±3.0 (manual; env bound is ±inf) | 1 (ignored) |
| Pole angle | ±0.2095 rad / ±12° (termination threshold; env bound ±0.418) | 6 |
| Pole angular velocity | ±3.5 (manual; env bound is ±inf) | 12 |

Position and angle are clipped at the episode-termination thresholds
(`env.unwrapped.x_threshold`, `theta_threshold_radians`) because the observation-space
bounds are 2× wider, which leaves half the bins unreachable. Only the pole's angle and
angular velocity are resolved: 72 states instead of 5,184 for `(6, 6, 12, 12)`, so each
state is visited often enough to learn within 1,000 episodes.

**Q-table.** Shape `bins + (n_actions,)` = `(1, 1, 6, 12, 2)`; `q_table[s]` holds the
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
4. Termination-threshold bounds and `(1, 1, 6, 12)` bins. With the original
   `(6, 6, 12, 12)` bins and env bounds, Q-Learning never converged on any of 3 seeds
   (final 100-ep MA 52.4 ± 17.7). This config was picked by a small grid search over
   bounds × bins × α ∈ {0.1, 0.2} × ε-decay ∈ {0.99, 0.995}, on the same seeds 42–44
   used for evaluation (so results are somewhat optimistic).

## REINFORCE (`cartpole_rl.py`, section 3)

**Policy network.** `PolicyNetwork` takes the raw 4-D state (no discretization) through
`Linear(4, 128) → ReLU → Linear(128, 2) → softmax` to get π(a | s; θ).

**Exploration.** Actions are sampled from π via `Categorical`, so exploration comes from
the policy's own stochasticity — no ε schedule.

**Update.** Once per episode (Monte Carlo):
1. Compute discounted returns `G_t = r_t + γ G_{t+1}` (γ = 0.99) with `discounted_returns()`.
2. Normalize returns to zero mean / unit std — a simple baseline that reduces variance.
3. Loss `= −Σ_t log π(a_t | s_t) · G_t`, minimized with Adam (lr = 5e-3).

No bootstrapping, so truncation at 500 steps needs no special case.

`reinforce(env, ...)` returns `(rewards, losses, policy)`. It trains on a fresh seeded
environment (`make_env()`) with `torch.manual_seed(SEED)` reset, so both agents start
from the same conditions.

**Learning rate.** lr was lowered from 1e-2 to 5e-3: at 1e-2 the policy on seed 44
collapsed to always choosing one action within ~10 episodes (episodes of ~9 steps,
never recovered), and only 2 of 3 seeds converged.

## Training & evaluation (`cartpole_rl.py`, section 4)

Sections 2 and 3 now only define the agents; section 4 trains both. Each method is trained
for 1,000 episodes on each of 3 seeds (`SEEDS = [42, 43, 44]`). `run_method()` reseeds
Python/NumPy/PyTorch and builds a fresh seeded environment per run. Recorded per run:

- **Reward per episode** and the **100-episode moving average** (trailing; NaN until 100 episodes exist)
- **Convergence episode**: first episode where the 100-ep MA ≥ 195 (`None` if never)
- **Successful episodes**: reward ≥ 195 (this includes every episode that reaches the 500 cap)
- **Wall-clock training time** (`time.perf_counter`)

Everything is kept in `results[method]` (a list of per-seed dicts, including epsilons or
losses and the trained Q-table or policy) for plotting. A per-run log and a mean ± std
summary table are printed. Full run takes ~6 minutes.

**Current results** (mean ± std over 3 seeds; convergence shows how many seeds converged):

| Metric | Q-Learning | REINFORCE |
|---|---|---|
| Converged at episode | 475 ± 135 (3/3) | 132 ± 5 (3/3) |
| Successful episodes | 337 ± 59 | 802 ± 122 |
| Final 100-ep MA | 333.5 ± 162.2 | 424.3 ± 128.2 |
| Best episode | 500 ± 0 | 500 ± 0 |
| Training time (s) | 12.8 ± 0.7 | 106.6 ± 24.5 |

Per seed (42 / 43 / 44): Q-Learning converged at 580 / 322 / 522 (final MA 148.9 /
453.3 / 398.3); REINFORCE at 136 / 134 / 127 (final MA 496.6 / 276.3 / 500.0).
REINFORCE converges ~3.6× sooner in episodes but takes ~8× longer in wall-clock time
(a neural-network forward/backward pass per step vs. a table lookup, and longer episodes).

## Plots (`cartpole_rl.py`, section 5)

Seaborn figures saved to `figures/` at 200 dpi, sized for 16:9 slides (13.33 in wide).
Q-Learning is always blue, REINFORCE always orange.

| File | Contents |
|---|---|
| `learning_curves.png` | Reward per episode (faint) + 100-episode moving average (bold) for both methods, mean of 3 seeds, ±1 SD band, dashed "solved" line at 195 |
| `learning_curves_per_seed.png` | Same curves, one panel per seed — shows run-to-run variance and post-convergence dips |
| `metrics_comparison.png` | Three bar panels (convergence episode, successful episodes, training time): bar = mean, whisker = ±1 SD, dots = individual seeds. Separate panels because the units differ |
| `metrics_by_seed.csv` | Per-seed values behind the bar chart |

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
