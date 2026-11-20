# CST_675_Week3
Implement and Compare Q-Learning and REINFORCE in Python

## Goal
Compare a **value-based** method (tabular Q-Learning) with a **policy-based** method
(REINFORCE) on the CartPole-v1 environment from OpenAI Gym / Gymnasium.

**Objective:** compare learning speed, stability, and compute cost of the two approaches
to guide the choice of RL algorithm on future projects.

## Setup
Requires Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt # gymnasium, numpy, matplotlib, torch, seaborn
python cartpole_rl.py
```

The script uses **Gymnasium** (`import gymnasium as gym`), the maintained successor
to OpenAI Gym, with the 5-value `step()` API:
`obs, reward, terminated, truncated, info`. All RNGs (Python, NumPy, PyTorch, and the
environment) are seeded with `SEED = 42`, so runs are reproducible.

**CartPole-v1 at a glance:** 4-D continuous observation (cart position, cart velocity,
pole angle, pole angular velocity), 2 discrete actions (push left/right), +1 reward per
step, episodes capped at 500 steps. An episode terminates early if the pole tips past
±12° or the cart leaves ±2.4 m. The two velocity components are unbounded, so
Q-Learning needs clipped ranges for them when the state is discretized. The task counts
as "solved" when the 100-episode moving average reaches 195.

## Deliverables
- Working Python implementations of both agents (tabular Q-Learning + REINFORCE) — `cartpole_rl.py`
- Training metrics and plots — `figures/`
- Strengths/weaknesses analysis — [`ANALYSIS.md`](ANALYSIS.md)

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
`Linear(4, 128) → ReLU → Linear(128, 2) → softmax` to get π(a | s; θ) — 898 parameters.

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

| Derived metric (mean of 3 seeds) | Q-Learning | REINFORCE |
|---|---|---|
| Environment steps to convergence | ~51.7k | ~20.4k |
| Compute per environment step | ~64 µs | ~277 µs |

## Plots (`cartpole_rl.py`, section 5)

Seaborn figures saved to `figures/` at 200 dpi, sized for 16:9 slides (13.33 in wide).
Q-Learning is always blue, REINFORCE always orange.

| File | Contents |
|---|---|
| `learning_curves.png` | Reward per episode (faint) + 100-episode moving average (bold) for both methods, mean of 3 seeds, ±1 SD band, dashed "solved" line at 195 |
| `learning_curves_per_seed.png` | Same curves, one panel per seed — shows run-to-run variance and post-convergence dips |
| `metrics_comparison.png` | Three bar panels (convergence episode, successful episodes, training time): bar = mean, whisker = ±1 SD, dots = individual seeds. Separate panels because the units differ |
| `metrics_by_seed.csv` | Per-seed values behind the bar chart |

## Key Findings

- **3.6× fewer episodes to converge with REINFORCE** (132 ± 5 vs. 475 ± 135), and ~2.5×
  fewer environment steps, with a much tighter spread across seeds.
- **~4× lower compute per step with Q-Learning** (table lookup vs. network forward/backward
  pass), so measured in compute time Q-Learning reached the solved bar first (~3.3 s vs. ~5.7 s).
- **13 of 40 binning configurations converged on every seed** (14 converged on none):
  the state representation, not the learning rule, decided whether Q-Learning worked.
- **Both methods can lose a good policy late in training.** Q-Learning's moving average fell
  back below 195 for 14–43% of post-convergence episodes; REINFORCE dropped late on seed 43.

## Strengths, Weaknesses & Applicability

| | Q-Learning (value-based) | REINFORCE (policy-based) |
|---|---|---|
| State space | − Must discretize; table grows exponentially (72 states worked, 5,184 failed) | + Raw continuous state through a neural network; generalizes |
| Sample efficiency | − ~51.7k steps here: no replay, coarse table generalizes poorly | + ~20.4k steps here; − discards each episode after one update |
| Variance | + Low-variance TD updates; − bootstrap bias, wide seed spread (±135) | − High-variance Monte Carlo gradient; + tight seed spread (±5) |
| Stability | − Fell below 195 for 14–43% of post-convergence episodes | + Mostly stable; − one late collapse, learning-rate sensitive |
| On/off-policy | Off-policy: learns greedy policy from ε-greedy data | On-policy: only data from the current policy is valid |
| Implementation | + ~40 lines of NumPy, inspectable table; − bin/bounds tuning | − Needs autograd, more concepts; + no feature engineering |
| Best fit | Small, discrete tasks: grids, inventory, queues | Continuous actions, high-dimensional inputs |
| Successor | DQN: neural Q-function + replay buffer + target network | Actor-critic / PPO: critic baseline cuts variance; clipped updates |

Full discussion with in-text citations: [`ANALYSIS.md`](ANALYSIS.md).

## Recommendation

- Use tabular Q-Learning only for small, discrete problems; keep plain REINFORCE as a baseline.
- Start from a maintained library (Stable-Baselines3): **DQN** for discrete actions with
  costly data, **PPO** as the general default (including continuous actions).
- Evaluate on 5+ seeds, tune on held-out seeds, and ship the best checkpoint, not the last.

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

## References

Churchill, D. (2025, March 25). *COMP 3200 / 6980 - Intro to artificial intelligence - Lecture 19 - Temporal difference learning + A5* [Video]. YouTube. https://www.youtube.com/watch?v=1R73DMnMe5s

GeeksforGeeks. (2024, June 17). *Differences between Q-learning and SARSA*. https://www.geeksforgeeks.org/differences-between-q-learning-and-sarsa/

Huang, Z. (2025, August 29). *Reinforcement learning #4: Temporal-difference learning, Q-learning, SARSA* [Video]. YouTube. https://www.youtube.com/watch?v=HSp5nnf3fgY

Kapoor, S. (2018, June 21). Policy gradients in a nutshell. *Towards Data Science*. https://medium.com/data-science/policy-gradients-in-a-nutshell-8b72f9743c5d

Luu, Q. T. (2024, March 18). *Q-learning vs. deep Q-learning vs. deep Q-network*. Baeldung on Computer Science. https://www.baeldung.com/cs/q-learning-vs-deep-q-learning-vs-deep-q-network

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533. https://doi.org/10.1038/nature14236

Mutual Information. (2023, May 3). *Policy gradient methods | Reinforcement learning part 6* [Video]. YouTube. https://www.youtube.com/watch?v=e20EY4tFC_Q

Or, B. (2021, January 29). Value-based methods in deep reinforcement learning. *Towards Data Science*. https://medium.com/data-science/value-based-methods-in-deep-reinforcement-learning-d40ca1086e1

Papers in 100 Lines of Code. (n.d.). *Value-based vs policy-based reinforcement learning*. Medium. https://papers-100-lines.medium.com/value-based-vs-policy-based-reinforcement-learning-92da766696fd

PyTorch. (n.d.). *Reinforcement learning (DQN) tutorial*. PyTorch Tutorials. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. *Journal of Machine Learning Research, 22*(268), 1–8. https://jmlr.org/papers/v22/20-1364.html

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms* (arXiv:1707.06347). arXiv. https://doi.org/10.48550/arXiv.1707.06347

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Tech Entertaining. (2023, August 25). *19. State value & action value function || End to end AI tutorial* [Video]. YouTube. https://www.youtube.com/watch?v=1j7kJZnbiF4

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning, 8*(3–4), 229–256. https://doi.org/10.1007/BF00992696

See [`ANALYSIS.md`](ANALYSIS.md) for the full reference list, including the O'Reilly course materials.
