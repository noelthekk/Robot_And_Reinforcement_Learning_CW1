from matplotlib.projections import HammerAxes
import numpy as np
from my_cartpole_env import CartPoleEnv
from gymnasium.wrappers import TimeLimit
from hyperparameters import *
from common import *
from plotting import plot_returns


class CartpoleMPC:
    def __init__(self, H=10, max_iters=5):

        # MPC/iLQR Parameters
        self.H = H          # Horizon length
        self.max_iters = max_iters    # iLQR iterations per time step
        ### FILL IN HERE ### hint: Q, R from provided cost function parameters
        # self.Q = np.diag([...])
        # self.R = np.array([[...]])
        self.Q = np.diag([1.0, 0.1, 10.0, 0.1])
        self.R = np.array([[0.01]])  # No control cost for simplicity, can be tuned if needed
        
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
    
    def stage_cost(self, x, u):
        # x = [pos, vel, theta, theta_dot]
        pos, vel, th, thd = x
        th_tar = np.deg2rad(40.0)

        # weights (start here, then tune)
        wx, wv = 0.1, 0.05
        wth = 5.0
        wthd = 0.5
        wu = 0.001

        # double-well in theta: minima at th = ±th_tar
        c_theta = wth * (th*th - th_tar*th_tar)**2

        # encourage motion (negative cost for angular speed)
        c_motion = -wthd * (thd*thd)

        # keep cart from drifting too much
        c_cart = wx * pos*pos + wv * vel*vel

        # control effort
        c_u = wu * float(u[0])**2

        return c_cart + c_theta + c_motion + c_u

    def terminal_cost(self, x):
        # keep terminal cost simple (don’t encourage high speed at the end)
        pos, vel, th, thd = x
        th_tar = np.deg2rad(40.0)

        wx, wv = 1.0, 0.2
        wth = 10.0
        wthd = 0.2

        return (
            wx * pos*pos
            + wv * vel*vel
            + wth * (th*th - th_tar*th_tar)**2
            + wthd * thd*thd
        )

    def cost_derivs_fd(self, x, u, eps=1e-4):
        # returns lx, lu, lxx, luu, lux using central differences
        nx, nu = len(x), len(u)

        l0 = self.stage_cost(x, u)

        lx = np.zeros(nx)
        for i in range(nx):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps; xm[i] -= eps
            lx[i] = (self.stage_cost(xp, u) - self.stage_cost(xm, u)) / (2*eps)

        lu = np.zeros(nu)
        for i in range(nu):
            up, um = u.copy(), u.copy()
            up[i] += eps; um[i] -= eps
            lu[i] = (self.stage_cost(x, up) - self.stage_cost(x, um)) / (2*eps)

        lxx = np.zeros((nx, nx))
        for i in range(nx):
            for j in range(nx):
                xpp = x.copy(); xpp[i]+=eps; xpp[j]+=eps
                xpm = x.copy(); xpm[i]+=eps; xpm[j]-=eps
                xmp = x.copy(); xmp[i]-=eps; xmp[j]+=eps
                xmm = x.copy(); xmm[i]-=eps; xmm[j]-=eps
                lxx[i,j] = (self.stage_cost(xpp,u)-self.stage_cost(xpm,u)-self.stage_cost(xmp,u)+self.stage_cost(xmm,u)) / (4*eps*eps)

        luu = np.zeros((nu, nu))
        for i in range(nu):
            for j in range(nu):
                upp = u.copy(); upp[i]+=eps; upp[j]+=eps
                upm = u.copy(); upm[i]+=eps; upm[j]-=eps
                ump = u.copy(); ump[i]-=eps; ump[j]+=eps
                umm = u.copy(); umm[i]-=eps; umm[j]-=eps
                luu[i,j] = (self.stage_cost(x,upp)-self.stage_cost(x,upm)-self.stage_cost(x,ump)+self.stage_cost(x,umm)) / (4*eps*eps)

        lux = np.zeros((nu, nx))
        for i in range(nu):
            for j in range(nx):
                up = u.copy(); um = u.copy()
                xp = x.copy(); xm = x.copy()
                up[i]+=eps; um[i]-=eps
                xp[j]+=eps; xm[j]-=eps
                lux[i,j] = (self.stage_cost(xp,up)-self.stage_cost(xm,up)-self.stage_cost(xp,um)+self.stage_cost(xm,um)) / (4*eps*eps)

        return lx, lu, lxx, luu, lux

    def term_derivs_fd(self, x, eps=1e-4):
        nx = len(x)
        Vx = np.zeros(nx)
        for i in range(nx):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps; xm[i] -= eps
            Vx[i] = (self.terminal_cost(xp) - self.terminal_cost(xm)) / (2*eps)

        Vxx = np.zeros((nx, nx))
        for i in range(nx):
            for j in range(nx):
                xpp = x.copy(); xpp[i]+=eps; xpp[j]+=eps
                xpm = x.copy(); xpm[i]+=eps; xpm[j]-=eps
                xmp = x.copy(); xmp[i]-=eps; xmp[j]+=eps
                xmm = x.copy(); xmm[i]-=eps; xmm[j]-=eps
                Vxx[i,j] = (self.terminal_cost(xpp)-self.terminal_cost(xpm)-self.terminal_cost(xmp)+self.terminal_cost(xmm)) / (4*eps*eps)

        return Vx, Vxx

    def solve_ilqr(self, x0, U_init):
        """The iLQR Solver"""
        U = U_init.copy()
        X = np.zeros((self.H + 1, 4))
        X[0] = x0
        
        # Initial Rollout
        for t in range(self.H):
            X[t+1] = dynamics(X[t], U[t], continuous_action=True)
            
        for _ in range(self.max_iters):
            # Backward Pass
            ks = [np.zeros((1, 1))] * self.H
            Ks = [np.zeros((1, 4))] * self.H
            
            # Terminal Value Function derivatives
            Vx, Vxx = self.term_derivs_fd(X[-1])

            for t in reversed(range(self.H)):
                A, B = self.get_jacobians(X[t], U[t])

                # Gradients of the cost
                lx, lu, lxx, luu, lux = self.cost_derivs_fd(X[t], U[t])

                ### FILL IN HERE ### hint: Q-function derivatives, control gains, value function update
                Qx  = lx  + A.T @ Vx
                Qu  = lu  + B.T @ Vx
                Qxx = lxx + A.T @ Vxx @ A
                Quu = luu + B.T @ Vxx @ B
                Qux = lux + B.T @ Vxx @ A

                # Compute control gains (assuming Quu is invertible)
                Quu = Quu + 1e-6 * np.eye(Quu.shape[0])
                Quu_inv = np.linalg.inv(Quu)
                ks[t] = -Quu_inv @ Qu
                Ks[t] = -Quu_inv @ Qux 

                # Update value function derivatives for next iteration
                Vx = Qx + Ks[t].T @ Quu @ ks[t] + Ks[t].T @ Qu + Qux.T @ ks[t]
                Vxx = Qxx + Ks[t].T @ Quu @ Ks[t] + Ks[t].T @ Qux + Qux.T @ Ks[t]

            # Forward Pass (Line search simplified for brevity)
            X_new = np.zeros_like(X)
            X_new[0] = x0
            U_new = np.zeros_like(U)
            
            for t in range(self.H):
                ### FILL IN HERE ### hint: compute U_new[t] and X_new[t+1]
                alpha = 0.2
                U_new[t] = U[t] + alpha * ks[t] + Ks[t] @ (X_new[t] - X[t])
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
    USE_MPC = True  # Set to False for Open-Loop iLQR
    DISTURBANCE = 0
    H_MPC = 15
    H_ILQR = 100
    # ----------------------
    # MAX_EPISODE_STEPS =5

    env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT, continuous_action=True, disturbance=DISTURBANCE)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    play_env = CartPoleEnv(render_mode="human", x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT, continuous_action=True, disturbance=DISTURBANCE)
    play_env = TimeLimit(play_env, max_episode_steps=MAX_EPISODE_STEPS)

    if USE_MPC:
        print("Running in Closed-Loop MPC mode...")

        mpc = CartpoleMPC(H_MPC, max_iters=5)
        # returns = evaluate_agent(env, type="MPC", policy=mpc)
        # print_statistics(returns)
        play_agent(play_env, type="MPC", policy=mpc)
    else:
        print("Running in Open-Loop iLQR mode...")

        mpc = CartpoleMPC(H_ILQR, max_iters=5)
        state, _ = play_env.reset()
            
        while True:
            # Solve for the entire horizon
            u_plan, x_plan, ks, Ks = mpc.solve_ilqr(state, np.zeros((H_ILQR, 1)))
            
            for t in range(H_ILQR):
                action = u_plan[t] + ks[t] + Ks[t] @ (state - x_plan[t]) 
                state, _, terminated, truncated, info = play_env.step(action[0])
            
                if terminated or truncated:
                    state, _ = play_env.reset()
                    reason = "Terminated (Fell/Out of Bounds)" if terminated else "Truncated (Time Limit)"
                    print(f"Finished at t={t} | Reason: {reason}")
                    break
    
