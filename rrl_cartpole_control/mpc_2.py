from matplotlib.projections import HammerAxes
import numpy as np
from my_cartpole_env import CartPoleEnv
from gymnasium.wrappers import TimeLimit
from hyperparameters import *
from common import *
from plotting import plot_returns
import matplotlib.pyplot as plt


class CartpoleMPC:
    def __init__(self, H=10, max_iters=5):

        # MPC/iLQR Parameters
        self.H = H          # Horizon length
        self.max_iters = max_iters    # iLQR iterations per time step
        ### FILL IN HERE ### hint: Q, R from provided cost function parameters
        # self.Q = np.diag([...])
        # self.R = np.array([[...]])
        self.Q = np.diag([1.0, 0.1, 10.0, 0.1])
        self.R = np.array([[0.01]])
        
        # Warm start buffer
        self.U_guess = np.zeros((self.H, 1))

    def get_jacobians(self, x, u):
        """Numerical Jacobians (Finite Difference) for A and B"""
        eps = 1e-6
        nx = len(x)
        nu = len(u)
        
        A = np.zeros((nx, nx))
        B = np.zeros((nx, nu))
        
        for i in range(nx):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            A[:, i] = (dynamics(x_plus, u, continuous_action=True) - dynamics(x_minus, u, continuous_action=True)) / (2 * eps)
            
        for i in range(nu):
            u_plus = u.copy()
            u_plus[i] += eps
            u_minus = u.copy()
            u_minus[i] -= eps
            B[:, i] = (dynamics(x, u_plus, continuous_action=True) - dynamics(x, u_minus, continuous_action=True)) / (2 * eps)
            
        return A, B

    def solve_ilqr(self, x0, U_init):
        """The iLQR Solver"""
        U = U_init.copy()
        X = np.zeros((self.H + 1, 4))
        X[0] = x0
        
        # Initial Rollout
        for t in range(self.H):
            X[t+1] = dynamics(X[t], U[t], continuous_action=True)
            
        for i in range(self.max_iters):
            # print(i)
            # Backward Pass
            ks = [np.zeros((1,1)) for _ in range(self.H)]
            Ks = [np.zeros((1,4)) for _ in range(self.H)]
            
            # Terminal Value Function derivatives
            Vx = self.Q @ X[-1]
            Vxx = self.Q
            
            for t in reversed(range(self.H)):
                A, B = self.get_jacobians(X[t], U[t])

                # Gradients of the cost
                lx = self.Q @ X[t]
                lu = self.R @ U[t]

                ### FILL IN HERE ### hint: Q-function derivatives, control gains, value function update
                
                Qx = lx + A.T @ Vx
                Qu = lu + B.T @ Vx
                Qxx = self.Q + A.T @ Vxx @ A
                Quu = self.R + B.T @ Vxx @ B
                Qux = B.T @ Vxx @ A

                # Compute control gains (K, k)
                K = -np.linalg.solve(Quu, Qux)
                k = -np.linalg.solve(Quu, Qu)

                ks[t] = k
                Ks[t] = K

                # Update value function derivatives
                Vx = Qx - K.T @ Quu @ k
                Vxx = Qxx - K.T @ Quu @ K

            # Forward Pass (Line search simplified for brevity)
            X_new = np.zeros_like(X)
            X_new[0] = x0
            U_new = np.zeros_like(U)
            
            for t in range(self.H):
                ### FILL IN HERE ### hint: compute U_new[t] and X_new[t+1]
                alpha = 0.1
                U_new[t] = U[t] + alpha*ks[t] + Ks[t] @ (X_new[t] - X[t])
                X_new[t+1] = dynamics(X_new[t], U_new[t], continuous_action=True)
            
            X, U = X_new, U_new

        ks = np.array(ks)
        Ks = np.array(Ks)
        return U, X, ks, Ks

    def reset(self):
        """Reset the warm start buffer for a new episode"""
        self.U_guess = np.zeros((self.H, 1))
    
    def control(self, state):
        """MPC interface: solve and shift"""
        U_opt, _, _, _ = self.solve_ilqr(state, self.U_guess)
        
        # Extract first action and ensure it's a scalar
        action = U_opt[0, 0] if U_opt.ndim == 2 else U_opt[0]
        
        # Clip action to valid range [-1, 1]
        action = float(np.clip(action, -1.0, 1.0))
        
        # Warm start shift
        self.U_guess[:-1] = U_opt[1:]
        self.U_guess[-1] = 0
        
        return action


if __name__ == "__main__":

    # ----------------------
    USE_MPC = False # Set to False for Open-Loop iLQR
    DISTURBANCE = 0
    H_MPC = 25
    H_ILQR = 100
    # ----------------------


    disturbance_levels = np.arange(0, 21, 1)  # 0 to 10
    mpc_term_rates = []
    ilqr_term_rates = []

    # N = NUM_EPISODES_EVAL  # how many episodes per disturbance
    N=20

    for disturbance in disturbance_levels:
        print(f"\nDisturbance={disturbance:.1f}")

        # IMPORTANT: recreate envs with the new disturbance
        env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT,
                        continuous_action=True, disturbance=disturbance)
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

        play_env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT,
                            continuous_action=True, disturbance=disturbance)
        play_env = TimeLimit(play_env, max_episode_steps=MAX_EPISODE_STEPS)

        # ---------------- MPC ----------------
        mpc = CartpoleMPC(H_MPC, max_iters=5)
        _, term_rate = evaluate_agent(env, type="MPC", policy=mpc)
        print(f"MPC Termination Rate: {term_rate:.2f}")
        mpc_term_rates.append(term_rate)


    # -------- plot both on same graph --------
    plt.figure()
    plt.plot(disturbance_levels, mpc_term_rates, label="MPC")
    # plt.plot(disturbance_levels, ilqr_term_rates, label="iLQR")
    plt.xlabel("Disturbance")
    plt.ylabel("Termination rate")
    plt.title("Termination rate vs disturbance - MPC")
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend()
    plt.savefig("termination_rate_vs_disturbance_mpc.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    disturbance_levels = np.arange(0, 201, 10)  # 0 to 10

    for disturbance in disturbance_levels:
        print(f"\nDisturbance={disturbance:.1f}")

        # IMPORTANT: recreate envs with the new disturbance
        env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT,
                        continuous_action=True, disturbance=disturbance)
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

        play_env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT,
                            continuous_action=True, disturbance=disturbance)
        play_env = TimeLimit(play_env, max_episode_steps=MAX_EPISODE_STEPS)


        # ---------------- iLQR (plan once per episode) ----------------
        ilqr = CartpoleMPC(H_ILQR, max_iters=1000)
        term_count = 0

        for ep in range(N):
            state, _ = play_env.reset()

            u_plan, x_plan, ks, Ks = ilqr.solve_ilqr(state, np.zeros((H_ILQR, 1)))

            terminated = False
            truncated = False
            for t in range(H_ILQR):
                action = u_plan[t]  + ks[t] + Ks[t] @ (state - x_plan[t])
                state, _, terminated, truncated, _ = play_env.step(action[0])
                if terminated or truncated:
                    print(f"Episode {ep+1}/{N} finished at t={t} | Reason: {'Terminated' if terminated else 'Truncated'}")
                    break

            if terminated:   # count real failures
                term_count += 1
        print(f"iLQR Termination Rate: {term_count / N:.2f}")
        ilqr_term_rates.append(term_count / N)

    # -------- plot both on same graph --------
    plt.figure()
    # plt.plot(disturbance_levels, mpc_term_rates, label="MPC")
    plt.plot(disturbance_levels, ilqr_term_rates, label="iLQR")
    plt.xlabel("Disturbance")
    plt.ylabel("Termination rate")
    plt.title("Termination rate vs disturbance - iLQR")
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend()
    plt.savefig("termination_rate_vs_disturbance_ilqr.pdf", dpi=300, bbox_inches='tight')
    plt.show()

