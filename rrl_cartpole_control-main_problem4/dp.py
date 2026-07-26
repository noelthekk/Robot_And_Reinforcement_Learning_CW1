import numpy as np
from gymnasium.wrappers import TimeLimit
from my_cartpole_env import CartPoleEnv
from plotting import plot_policy_value
from common import *
from hyperparameters import *


def value_iteration():
    global V

    # Pre-calculate the current state costs for efficiency
    current_state_costs = quadratic_cost(state_tensor)

    for it in range(N_ITERATIONS):
        V_old = V.copy()
        V_new = []

        for action in ACTION_VALS:
            next_states = dynamics(state_tensor, action)
            next_indices = state_to_indices(next_states, [x_vals, x_dot_vals, theta_vals, theta_dot_vals])
            future_val = current_state_costs + GAMMA * V_old[next_indices]
            new_estimate = np.where(is_terminal[next_indices], TERMINAL_COST, future_val)
            V_new.append(new_estimate)

        ### FILL IN HERE ### 
        V = np.min(V_new, axis=0)

        ### FILL IN HERE ###
        diff = np.max(np.abs(V_old - V))

        print(f"[{it}] diff = {diff:.5f}", "value stats: ", np.min(V), np.max(V), np.mean(V), end='\r')
        if diff < DELTA: print(f"VI completed in {it} iterations with diff {diff:.5f}"); break
    return V

def compute_policy(V):
    current_cost = quadratic_cost(state_tensor)

    q_vals = []
    for action in ACTION_VALS:
        ns   = dynamics(state_tensor, action)
        idx  = state_to_indices(ns, [x_vals, x_dot_vals, theta_vals, theta_dot_vals])
        term = is_terminal[idx]
        fv   = np.where(term, TERMINAL_COST, V[idx])
        q_vals.append(current_cost + GAMMA * fv)

    q0, q1 = q_vals[0], q_vals[1]

    # don't cast to int
    policy = np.argmin(np.stack([q0, q1], axis=0), axis=0).astype(float)

    tie = np.isclose(q0, q1, rtol=1e-5, atol=1e-8)

    # stats only for xdot≈0 and thetadot≈0 (in the full 4D policy)
    xdot0_idx = np.argmin(np.abs(x_dot_vals - 0.0))
    tdot0_idx = np.argmin(np.abs(theta_dot_vals - 0.0))

    # NOTE: policy isn't defined yet if you print before creating it,
    # so compute it first:
    policy = np.argmin(np.stack([q0, q1], axis=0), axis=0).astype(float)

    pol_00 = policy[:, xdot0_idx, :, tdot0_idx]
    tie_00 = tie[:, xdot0_idx, :, tdot0_idx]

    print("tie fraction (xdot≈0, thetadot≈0):", tie_00.mean())

    non_tie = ~tie_00
    if non_tie.any():
        print("a0 fraction (non-tie, xdot≈0, thetadot≈0):", (pol_00[non_tie] == 0).mean())
        print("a1 fraction (non-tie, xdot≈0, thetadot≈0):", (pol_00[non_tie] == 1).mean())
    else:
        print("No non-tie states in this subset.")

    policy[tie] = 0.5
    return policy


if __name__ == "__main__":

    # value iteration
    V = value_iteration()
    policy = compute_policy(V)
    plot_policy_value(policy, V, N_BINS, x_vals, theta_vals)

    # evaluate agent
    env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    returns = evaluate_agent(env, type="DP", policy=policy)
    print_statistics(returns)
    env.close()

    # play agent
    # play_env = CartPoleEnv(render_mode="human", x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT)
    # play_env = TimeLimit(play_env, max_episode_steps=MAX_EPISODE_STEPS)
    # play_agent(play_env, type="DP", policy=policy)
    # play_env.close()


