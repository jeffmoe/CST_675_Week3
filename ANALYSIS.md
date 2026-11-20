# Q-Learning vs. REINFORCE on CartPole-v1: Comparative Analysis

## 1. Summary

This project compared a value-based method, tabular Q-Learning, with a policy-based
method, REINFORCE, on Gymnasium's CartPole-v1. Each method trained for 1,000 episodes on
3 seeds (42, 43, 44). After tuning, both methods reached the "solved" bar (a 100-episode
moving average of at least 195) on every seed. They got there in very different ways.

| Metric (mean ± SD, 3 seeds) | Q-Learning | REINFORCE |
|---|---|---|
| Converged at episode | 475 ± 135 (3/3) | **132 ± 5** (3/3) |
| Environment steps to convergence | ~51,700 | **~20,400** |
| Successful episodes (reward ≥ 195) | 337 ± 59 | **802 ± 122** |
| Final 100-episode moving average | 333.5 ± 162.2 | **424.3 ± 128.2** |
| Training time, 1,000 episodes | **~12 s** | ~105 s |
| Compute per environment step | **~64 µs** | ~277 µs |

In short, REINFORCE learned faster and more consistently in both episodes and environment
steps. Q-Learning was about 4× cheaper per step. It took heavy hand-tuning of the state
discretization before it converged at all, and it was less stable after convergence.
Figures are in `figures/` (`learning_curves.png`, `learning_curves_per_seed.png`,
`metrics_comparison.png`).

## 2. Background

Q-Learning learns an action-value function *Q(s, a)*, the expected return of taking
action *a* in state *s* (Tech Entertaining, 2023). It acts by choosing the highest-valued
action, with ε-greedy exploration. It updates *Q* after every step with a temporal-difference
(TD) target that bootstraps from its own estimate of the next state, *r + γ max Q(s′, a′)*
(Huang, 2025; Sutton & Barto, 2018). REINFORCE instead learns a parameterized policy
π(a | s; θ) directly. After each full episode, it raises the log-probability of each action
in proportion to the return that followed (Williams, 1992; Kapoor, 2018; Mutual Information,
2023). In other words, value-based methods ask "how good is each action?" and
policy-based methods ask "which way should the action probabilities move?" (Papers in 100
Lines of Code, n.d.; Or, 2021).

## 3. Strengths and Weaknesses

### 3.1 Discretization and the curse of dimensionality

Tabular Q-Learning needs a finite state space, so CartPole's four continuous state values
must be split into bins. This was the single biggest source of difficulty in this project:

- **The starter discretization failed.** With `(6, 6, 12, 12)` bins (5,184 states) and the
  environment's reported bounds, Q-Learning never converged on any seed (final moving
  average 52.4 ± 17.7). Two velocity bounds were reported as ±∞, which makes equal-width
  bins meaningless. The position and angle bounds were twice the range in which an episode
  can actually continue, so half of those bins could never be visited.
- **Results were very sensitive to the binning choice.** In a 40-configuration grid search
  over bounds, bin counts, learning rate, and ε-decay, only 13 configurations converged on
  all 3 seeds, and 14 converged on none.
- **The best table threw information away.** The winning configuration, `(1, 1, 6, 12)`,
  ignores cart position and cart velocity entirely. It works because a 72-state table gets
  visited often enough to learn within 1,000 episodes. A 5,184-state table does not.

This is the curse of dimensionality in miniature. Table size grows exponentially with the
number of state dimensions. Ten bins per dimension is 10⁴ cells for CartPole but 10²⁰ for a
20-dimensional robot state, and most cells would never be visited (Luu, 2024; Sutton &
Barto, 2018). A table also cannot generalize: what is learned in one bin says nothing
about its neighbors.

REINFORCE avoids all of this. Its 898-parameter policy network reads the raw four-value
state, needs no bounds or bins, and generalizes smoothly across similar states (Kapoor,
2018). Function approximation is the standard answer to large or continuous state spaces
(Luu, 2024; Habib, n.d.-a).

### 3.2 Sample efficiency

Here REINFORCE was the more sample-efficient method. It needed about 20,400 environment
steps to converge versus about 51,700 for Q-Learning, roughly 2.5× fewer, and it did so with
far less spread across seeds. That runs against the usual rule of thumb that off-policy
value methods are more sample-efficient than on-policy policy gradients (Papers in 100 Lines
of Code, n.d.). The explanation is that tabular Q-Learning never uses the advantages that
rule assumes. It has no experience replay, so each transition is used once, and the
coarse table generalizes poorly. The rule of thumb applies to value methods such as DQN
that do reuse data (see §4.2).

Wall-clock cost tells a different story. Q-Learning's table lookup costs about 64 µs per
step, versus about 277 µs for a forward and backward pass through REINFORCE's network.
Because of that cheaper step, Q-Learning reached convergence in about 3.3 s of compute
versus about 5.7 s for REINFORCE. REINFORCE's much longer total training time (~105 s vs.
~12 s) comes mostly from success: its episodes ran to the 500-step cap, so it simulated
about 378,000 steps against Q-Learning's 192,000.

### 3.3 Variance

REINFORCE's gradient estimate is unbiased but high-variance. It is built from whole-episode
Monte Carlo returns, which can swing widely from one episode to the next (Kapoor, 2018;
Sutton & Barto, 2018). Normalizing returns within each episode, which acts as a simple
baseline, was essential. Even so, the learning rate turned out to be fragile: at lr = 1e-2,
the seed-44 policy collapsed to always pushing one direction within about 10 episodes and
never recovered. Halving the rate to 5e-3 fixed it.

Q-Learning's TD updates are lower-variance because each update depends on a single
transition, but bootstrapping from its own estimates adds bias (Huang, 2025). In this
project, Q-Learning's variance showed up between seeds instead of between updates: its
convergence episode ranged from 322 to 580 (SD 135), whereas REINFORCE's ranged from 127 to
136 (SD 5).

### 3.4 Stability

Neither method improves steadily; both can lose a good policy after learning it
(`learning_curves_per_seed.png`).

- **Q-Learning was the less stable method here.** After first converging, its moving
  average fell back below 195 for 14–43% of the remaining episodes, depending on the seed.
  On seed 42 it peaked at 381 and ended at 149. Coarse bins lump together states that need
  different actions, so updates in one part of a bin can undo what was learned in another.
- **REINFORCE stayed above the bar on two of three seeds**, never or almost never dropping
  below 195 after converging. Seed 43, though, fell from a peak of 499 to a final 276. A
  single large policy-gradient step can move the policy far from a good region, which is a
  known weakness of plain policy gradients.

The practical lesson is to keep a copy of the best policy seen during training. The
policy at the end of training is not reliably the best one.

### 3.5 On-policy vs. off-policy learning

Q-Learning is **off-policy**. It learns the value of the greedy policy while it behaves
ε-greedily, because its target uses the best next action rather than the one actually taken
(GeeksforGeeks, 2024; Churchill, 2025, 01:32; Huang, 2025, 17:36). That is the key
difference from on-policy SARSA (AI Sciences, n.d.-b). In principle this lets
Q-Learning learn from any experience, including old or logged data.

REINFORCE is **on-policy**. Its gradient is only valid for data gathered by the current
policy, so each episode is used for one update and then discarded (Sutton & Barto, 2018).
Its exploration comes from sampling the stochastic policy, so it needs no ε schedule. In
exchange, it cannot learn from historical data or from another policy's behavior, which
matters for business problems where collecting new experience is costly.

### 3.6 Ease of implementation

| | Q-Learning | REINFORCE |
|---|---|---|
| Core code | ~40 lines of NumPy, no gradients (ShaunPX1, 2022) | ~50 lines of PyTorch, needs autograd |
| Concepts | Table, ε-greedy, TD update | Policy network, log-probabilities, returns, normalization |
| Hard part | Discretization design: bounds, bins (a 40-config search) | Learning-rate sensitivity; collapse at lr = 1e-2 |
| Edge cases | Must separate *terminated* (no bootstrap) from *truncated* (bootstrap) | None: Monte Carlo returns need no bootstrapping |
| Interpretability | High: the Q-table can be inspected directly | Low: a neural-network policy |

Q-Learning is easier to write, but the time saved was spent tuning the discretization.
REINFORCE takes more concepts up front but needed less problem-specific engineering.

## 4. Applicability

### 4.1 Where each fits best

**Tabular Q-Learning** fits small, discrete problems:
- Grid worlds, board positions, inventory levels, and queue lengths, where the state is
  naturally discrete and the table fits in memory
- Problems where interpretability and cheap computation matter
- Settings that benefit from off-policy learning, such as learning from logged data
- Teaching and baselines

It does not fit continuous or high-dimensional states (curse of dimensionality) or
continuous actions, because acting requires an *argmax* over a finite action set (Luu, 2024).

**REINFORCE**, and policy gradients in general, fits:
- Continuous action spaces, such as torque or steering angle, where a Gaussian policy
  replaces the softmax (Kapoor, 2018)
- High-dimensional or raw observations that need function approximation
- Problems where a stochastic policy is itself useful

Plain REINFORCE is rarely the right final choice, however. Its variance and on-policy data
use make it costly and brittle at scale, so it serves best as a baseline or teaching tool.

### 4.2 How DQN and actor-critic methods address each method's weaknesses

**DQN fixes Q-Learning's weaknesses.** It replaces the table with a neural network that
approximates *Q(s, a)*, which removes discretization and generalizes across states (Mnih
et al., 2015; Or, 2021; Habib, n.d.-a). Two additions keep training stable. An
*experience-replay buffer* reuses past transitions many times and breaks the correlation
between consecutive samples, which turns the off-policy property into real sample efficiency.
A *target network* is a lagged copy of the Q-network used to compute TD targets, so targets
don't shift with every update (PyTorch, n.d.; Habib, n.d.-b; AI Sciences, n.d.-a;
CodeEmporium, 2023). The PyTorch DQN tutorial solves this same CartPole task from raw
states (PyTorch, n.d.). DQN still needs a discrete action set.

**Actor-critic methods fix REINFORCE's weaknesses.** They add a learned value function, the
*critic*, which serves as a baseline and replaces the full Monte Carlo return with a
bootstrapped TD or advantage estimate. That sharply reduces gradient variance and allows
updates every step instead of every episode (Sutton & Barto, 2018; Mutual Information,
2023). Modern variants such as PPO also limit how far each update can move the policy
(Schulman et al., 2017). That directly targets the kind of post-convergence collapse
REINFORCE showed on seed 43. Actor-critic methods keep policy gradients' support for
continuous actions, and off-policy actor-critics recover replay-based sample efficiency.

The two families meet in the middle. DQN brings function approximation to value methods,
and actor-critic brings value estimates to policy methods.

## 5. Recommendation for Future Company Projects

1. **Don't use tabular Q-Learning or plain REINFORCE in production.** Use tabular
   Q-Learning only when the state and action spaces are small and naturally discrete, or
   when the policy must be fully inspectable. Keep plain REINFORCE as a baseline.
2. **Default to proven deep-RL algorithms from a maintained library such as Stable-Baselines3**
   (Raffin et al., 2021), rather than custom implementations. Both methods here needed
   tuning just to work on CartPole.
   - **Discrete actions with costly or limited data**, such as simulators that are slow
     or logged business data: use **DQN** (or variants such as Double DQN), which makes
     use of replay and off-policy learning.
   - **Continuous actions, or a general-purpose default**: use **PPO**, an actor-critic
     method that is robust to hyperparameters and supports discrete and continuous actions.
3. **Adopt the evaluation practices this project showed are necessary:**
   - Run at least 5 seeds and report mean ± SD. With 3 seeds, a single run could swing
     REINFORCE's average by over 100 points.
   - Tune on different seeds from those used for final evaluation. The Q-Learning
     settings here were picked on the evaluation seeds, so its results are somewhat
     optimistic.
   - Keep a copy of the best policy and evaluate it with exploration turned off, rather
     than shipping the final policy. Both methods lost performance late in training.
   - Track wall-clock time and environment steps separately. Which one is the real
     constraint decides between replay-based value methods and on-policy methods.
4. **Budget engineering time for representation.** Choosing the state features (or bins)
   decided whether Q-Learning learned at all, which mattered more than its learning rate
   or exploration schedule.

## 6. Limitations

- Only 3 seeds and 1,000 episodes per run. The standard deviations are wide, and the
  rankings are suggestive rather than statistically tested.
- The Q-Learning configuration was tuned on the same seeds used for evaluation, and
  REINFORCE's learning rate was chosen after seeing one seed fail. Both results are
  somewhat optimistic.
- Metrics come from training episodes with exploration still on. A separate test with
  exploration off was not run.
- Timings come from a single CPU machine and include Python overhead.
- CartPole is a small, low-dimensional benchmark, so conclusions about scaling rely on the
  literature rather than this experiment.

## References

AI Sciences. (n.d.-a). Chapter 9: Deep RL – DQN [Video course]. In *Reinforcement learning
and deep RL Python (theory and projects)*. O'Reilly Media.
https://learning.oreilly.com/course/reinforcement-learning-and/9781804610626/

AI Sciences. (n.d.-b). Introduction to SARSA [Video]. In *Reinforcement learning and deep RL
Python (theory and projects)*. O'Reilly Media.
https://learning.oreilly.com/videos/reinforcement-learning-and/9781804610626/9781804610626-video7_1/

Churchill, D. (2025, March 25). *COMP 3200 / 6980 - Intro to artificial intelligence -
Lecture 19 - Temporal difference learning + A5* [Video]. YouTube.
https://www.youtube.com/watch?v=1R73DMnMe5s

CodeEmporium. (2023, November 28). *Deep Q-networks explained!* [Video]. YouTube.
https://www.youtube.com/watch?v=x83WmvbRa2I

GeeksforGeeks. (2024, June 17). *Differences between Q-learning and SARSA*.
https://www.geeksforgeeks.org/differences-between-q-learning-and-sarsa/

Habib, N. (n.d.-a). Chapter 5: Building Q-networks with TensorFlow. In *Hands-on Q-learning
with Python*. O'Reilly Media.
https://learning.oreilly.com/library/view/hands-on-q-learning-with/9781789345803/eb6580f1-c237-4f08-8e24-719fe87f4417.xhtml

Habib, N. (n.d.-b). Chapter 6: Digging deeper into deep Q-networks with Keras and
TensorFlow. In *Hands-on Q-learning with Python*. O'Reilly Media.
https://learning.oreilly.com/library/view/hands-on-q-learning-with/9781789345803/eb1d5cbd-6524-4b8f-ab86-e305e33d0a5a.xhtml

Huang, Z. (2025, August 29). *Reinforcement learning #4: Temporal-difference learning,
Q-learning, SARSA* [Video]. YouTube. https://www.youtube.com/watch?v=HSp5nnf3fgY

Kapoor, S. (2018, June 21). Policy gradients in a nutshell. *Towards Data Science*.
https://medium.com/data-science/policy-gradients-in-a-nutshell-8b72f9743c5d

Luu, Q. T. (2024, March 18). *Q-learning vs. deep Q-learning vs. deep Q-network*. Baeldung
on Computer Science.
https://www.baeldung.com/cs/q-learning-vs-deep-q-learning-vs-deep-q-network

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves,
A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A.,
Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015).
Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533.
https://doi.org/10.1038/nature14236

Mutual Information. (2023, May 3). *Policy gradient methods | Reinforcement learning part 6*
[Video]. YouTube. https://www.youtube.com/watch?v=e20EY4tFC_Q

Or, B. (2021, January 29). Value-based methods in deep reinforcement learning. *Towards Data
Science*. https://medium.com/data-science/value-based-methods-in-deep-reinforcement-learning-d40ca1086e1

Papers in 100 Lines of Code. (n.d.). *Value-based vs policy-based reinforcement learning*.
Medium. https://papers-100-lines.medium.com/value-based-vs-policy-based-reinforcement-learning-92da766696fd

PyTorch. (n.d.). *Reinforcement learning (DQN) tutorial*. PyTorch Tutorials.
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021).
Stable-Baselines3: Reliable reinforcement learning implementations. *Journal of Machine
Learning Research, 22*(268), 1–8. https://jmlr.org/papers/v22/20-1364.html

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy
optimization algorithms* (arXiv:1707.06347). arXiv. https://doi.org/10.48550/arXiv.1707.06347

ShaunPX1. (2022, April 15). *Q-learning algorithm from scratch | Python Google Colab*
[Video]. YouTube. https://www.youtube.com/watch?v=-LD1Fj76bQg

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.).
MIT Press.

Tech Entertaining. (2023, August 25). *19. State value & action value function || End to
end AI tutorial* [Video]. YouTube. https://www.youtube.com/watch?v=1j7kJZnbiF4

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist
reinforcement learning. *Machine Learning, 8*(3–4), 229–256.
https://doi.org/10.1007/BF00992696
