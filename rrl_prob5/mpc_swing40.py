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

        # Keep these for reference (not used once we use shaped costs)
        self.Q = np.diag([1.0, 0.1, 10.0, 0.1])
        self.R = np.array([[0.01]])

        # For reference tracking (sine swing)
        self.dt = 0.02  # matches env Euler step in coursework
        self.th_amp = np.deg2rad(40.0)   # target swing amplitude
        self.period = 2.0               # seconds (tune: smaller -> faster oscillation)
        self.omega = 2.0 * np.pi / self.period

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

    def ref_theta(self, t):
        th_ref = self.th_amp * np.sin(self.omega * t)
        thd_ref = self.th_amp * self.omega * np.cos(self.omega * t)
        return th_ref, thd_ref

    def stage_cost(self, x, u, t):
        # x = [pos, vel, theta, theta_dot]
        pos, vel, th, thd = x
        th_ref, thd_ref = self.ref_theta(t)

        # weights (tune these)
        wx, wv = 0.2, 0.1
        wth, wthd = 30.0, 1.0
        wu = 0.002

        c_track = wth * (th - th_ref) ** 2 + wthd * (thd - thd_ref) ** 2
        c_cart = wx * pos ** 2 + wv * vel ** 2
        c_u = wu * float(u[0]) ** 2

        return c_track + c_cart + c_u

    def terminal_cost(self, x, t):
        # terminal tracking (slightly stronger)
        pos, vel, th, thd = x
        th_ref, thd_ref = self.ref_theta(t)

        wx, wv = 1.0, 0.2
        wth, wthd = 50.0, 2.0

        return (
            wx * pos ** 2
            + wv * vel ** 2
            + wth * (th - th_ref) ** 2
            + wthd * (thd - thd_ref) ** 2
        )

    def cost_derivs_fd(self, x, u, t, eps=1e-4):
        # returns lx, lu, lxx, luu, lux using central differences
        nx, nu = len(x), len(u)

        lx = np.zeros(nx)
        for i in range(nx):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            lx[i] = (self.stage_cost(xp, u, t) - self.stage_cost(xm, u, t)) / (2 * eps)

        lu = np.zeros(nu)
        for i in range(nu):
            up, um = u.copy(), u.copy()
            up[i] += eps
            um[i] -= eps
            lu[i] = (self.stage_cost(x, up, t) - self.stage_cost(x, um, t)) / (2 * eps)

        lxx = np.zeros((nx, nx))
        for i in range(nx):
            for j in range(nx):
                xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
                xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
                xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
                xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
                lxx[i, j] = (
                    self.stage_cost(xpp, u, t)
                    - self.stage_cost(xpm, u, t)
                    - self.stage_cost(xmp, u, t)
                    + self.stage_cost(xmm, u, t)
                ) / (4 * eps * eps)

        luu = np.zeros((nu, nu))
        for i in range(nu):
            for j in range(nu):
                upp = u.copy(); upp[i] += eps; upp[j] += eps
                upm = u.copy(); upm[i] += eps; upm[j] -= eps
                ump = u.copy(); ump[i] -= eps; ump[j] += eps
                umm = u.copy(); umm[i] -= eps; umm[j] -= eps
                luu[i, j] = (
                    self.stage_cost(x, upp, t)
                    - self.stage_cost(x, upm, t)
                    - self.stage_cost(x, ump, t)
                    + self.stage_cost(x, umm, t)
                ) / (4 * eps * eps)

        lux = np.zeros((nu, nx))
        for i in range(nu):
            for j in range(nx):
                up = u.copy(); um = u.copy()
                xp = x.copy(); xm = x.copy()
                up[i] += eps; um[i] -= eps
                xp[j] += eps; xm[j] -= eps
                lux[i, j] = (
                    self.stage_cost(xp, up, t)
                    - self.stage_cost(xm, up, t)
                    - self.stage_cost(xp, um, t)
                    + self.stage_cost(xm, um, t)
                ) / (4 * eps * eps)

        return lx, lu, lxx, luu, lux

    def term_derivs_fd(self, x, t, eps=1e-4):
        nx = len(x)

        Vx = np.zeros(nx)
        for i in range(nx):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            Vx[i] = (self.terminal_cost(xp, t) - self.terminal_cost(xm, t)) / (2 * eps)

        Vxx = np.zeros((nx, nx))
        for i in range(nx):
            for j in range(nx):
                xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
                xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
                xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
                xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
                Vxx[i, j] = (
                    self.terminal_cost(xpp, t)
                    - self.terminal_cost(xpm, t)
                    - self.terminal_cost(xmp, t)
                    + self.terminal_cost(xmm, t)
                ) / (4 * eps * eps)

        return Vx, Vxx

    def solve_ilqr(self, x0, U_init):
        """The iLQR Solver"""
        U = U_init.copy()
        X = np.zeros((self.H + 1, 4))
        X[0] = x0

        # Initial Rollout
        for k in range(self.H):
            X[k + 1] = dynamics(X[k], U[k], continuous_action=True)

        for _ in range(self.max_iters):
            ks = [np.zeros((1, 1))] * self.H
            Ks = [np.zeros((1, 4))] * self.H

            # terminal derivatives at time t=H*dt
            tH = self.H * self.dt
            Vx, Vxx = self.term_derivs_fd(X[-1], tH)

            for k in reversed(range(self.H)):
                t = k * self.dt
                A, B = self.get_jacobians(X[k], U[k])

                lx, lu, lxx, luu, lux = self.cost_derivs_fd(X[k], U[k], t)

                Qx = lx + A.T @ Vx
                Qu = lu + B.T @ Vx
                Qxx = lxx + A.T @ Vxx @ A
                Quu = luu + B.T @ Vxx @ B
                Qux = lux + B.T @ Vxx @ A

                # regularise and invert
                Quu = Quu + 1e-6 * np.eye(Quu.shape[0])
                Quu_inv = np.linalg.inv(Quu)

                ks[k] = -Quu_inv @ Qu
                Ks[k] = -Quu_inv @ Qux

                Vx = Qx + Ks[k].T @ Quu @ ks[k] + Ks[k].T @ Qu + Qux.T @ ks[k]
                Vxx = Qxx + Ks[k].T @ Quu @ Ks[k] + Ks[k].T @ Qux + Qux.T @ Ks[k]

            # Forward Pass (damped step)
            X_new = np.zeros_like(X)
            X_new[0] = x0
            U_new = np.zeros_like(U)

            alpha = 0.2
            for k in range(self.H):
                U_new[k] = U[k] + alpha * ks[k] + Ks[k] @ (X_new[k] - X[k])
                X_new[k + 1] = dynamics(X_new[k], U_new[k], continuous_action=True)

            X, U = X_new, U_new

        ks = np.array(ks)
        Ks = np.array(Ks)
        return U, X, ks, Ks

    def reset(self):
        self.U_guess = np.zeros((self.H, 1))

    def control(self, state):
        U_opt, _, _, _ = self.solve_ilqr(state, self.U_guess)

        action = U_opt[0, 0] if U_opt.ndim == 2 else U_opt[0]
        action = float(np.clip(action, -1.0, 1.0))

        self.U_guess[:-1] = U_opt[1:]
        self.U_guess[-1] = 0
        return action


if __name__ == "__main__":

    USE_MPC = True
    DISTURBANCE = 0
    H_MPC = 25
    H_ILQR = 100

    env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT, continuous_action=True, disturbance=DISTURBANCE)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    play_env = CartPoleEnv(render_mode="human", x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT, continuous_action=True, disturbance=DISTURBANCE)
    play_env = TimeLimit(play_env, max_episode_steps=MAX_EPISODE_STEPS)

    if USE_MPC:
        print("Running in Closed-Loop MPC mode...")
        mpc = CartpoleMPC(H_MPC, max_iters=5)
        play_agent(play_env, type="MPC", policy=mpc)
    else:
        print("Running in Open-Loop iLQR mode...")
        mpc = CartpoleMPC(H_ILQR, max_iters=100)
        state, _ = play_env.reset()

        while True:
            u_plan, x_plan, ks, Ks = mpc.solve_ilqr(state, np.zeros((H_ILQR, 1)))

            for t in range(H_ILQR):
                action = u_plan[t] + ks[t] + Ks[t] @ (state - x_plan[t])
                state, _, terminated, truncated, info = play_env.step(action[0])

                if terminated or truncated:
                    state, _ = play_env.reset()
                    reason = "Terminated (Fell/Out of Bounds)" if terminated else "Truncated (Time Limit)"
                    print(f"Finished at t={t} | Reason: {reason}")
                    break
