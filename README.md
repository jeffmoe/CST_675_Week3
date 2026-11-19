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
- [ ] Working Python implementations of both agents (setup section done)
- [ ] Training metrics and plots
- [ ] Strengths/weaknesses analysis
- [ ] 6–7 slide PowerPoint with speaker notes and APA references

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
