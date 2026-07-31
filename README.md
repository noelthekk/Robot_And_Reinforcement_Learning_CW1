# Robot and Reinforcement Learning — CW1

Coursework 1 for Robot and Reinforcement Learning, covering dynamic programming, model predictive control, and Q-learning on a cartpole environment.

This coursework studies the cartpole swing-up and stabilization problem through three complementary control paradigms: dynamic programming (value iteration over a discretized state space), model predictive control (receding-horizon optimization using a known dynamics model), and Q-learning (a model-free, tabular reinforcement learning approach). Across the problems, the state space is discretized over cart position, cart velocity, pole angle, and pole angular velocity, and a shared quadratic cost function penalizes cart displacement and pole angle while applying a terminal cost near the goal state. The aim is to compare how each method balances planning versus learning, sample efficiency, and robustness to discretization, and to characterize the trade-offs between exact model-based control and model-free trial-and-error learning on a classic underactuated control task.

## Structure

- `rrl_cartpole_control/` — main cartpole control notebooks and solution
- `rrl_cartpole_control-main_problem4/` — MPC and Q-learning scripts for problem 4
- `rrl_prob5/` — problem 5 scripts and analysis
- `common.py`, `hyperparameters.py`, `my_cartpole_env.py`, `plotting.py` — shared utilities used across the notebooks
