import numpy as np
import matplotlib.pyplot as plt
from common import dynamics
from my_cartpole_env import CartPoleEnv
from hyperparameters import X_LIMIT, THETA_LIMIT
# from mpc_swing_2 import CartpoleMPC  # adjust filename if different

# ---- Final tuned weights ----
wx  = 0.25
wv  = 0.1
wth = 100.0
wthd= 3.0
wu  = 0.001

H = 25
T = 500
DISTURBANCE = 0

class CartpoleMPC:
    def __init__(self, H=10, max_iters=5, wx=0.2, wv=0.1, wth=30.0, wthd=1.0, wu=0.002):  
        # MPC/iLQR Parameters
        self.H = H          # Horizon length
        self.max_iters = max_iters    # iLQR iterations per time step
        self.history = []

        # Keep these for reference (not used once we use shaped costs)
        self.Q = np.diag([1.0, 0.1, 10.0, 0.1])
        self.R = np.array([[0.01]])

        # Mapping weights for sensitivity analysis [cite: 130]
        self.wx = wx
        self.wv = wv
        self.wth = wth
        self.wthd = wthd
        self.wu = wu
        
        # For reference tracking (sine swing)
        self.dt = 0.02  # matches env Euler step in coursework
        self.th_amp = np.deg2rad(40.0)   # target swing amplitude
        self.period = 2.0               # seconds (tune: smaller -> faster oscillation)
        self.omega = 2.0 * np.pi / self.period

        # Episode time (so the reference doesn't restart every MPC solve)
        self.t0 = 0.0

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

        # # weights (tune these) HERE !!======================================
        # wx = 0.2 # cart position wght, more diff from x=0 
        # wv = 0.1 # cart velocity, to stop quick mov and make smooth
        # wthd = 1.0 # pole angular weight. smooth out motion
        # wth = 30.0 # pole angle weight, diff of actual angle and target angle
        # wu = 0.002

        # c_track = wth * (th - th_ref) ** 2 + wthd * (thd - thd_ref) ** 2
        # c_cart = wx * pos ** 2 + wv * vel ** 2
        # c_u = wu * float(u[0]) ** 2
        
        # Cost calculation using current variations [cite: 128]
        c_track = self.wth * (th - th_ref)**2 + self.wthd * (thd - thd_ref)**2
        c_cart = self.wx * pos**2 + self.wv * vel**2
        c_u = self.wu * float(u[0])**2

        return c_track + c_cart + c_u

    def terminal_cost(self, x, t):
        # terminal tracking (slightly stronger)
        pos, vel, th, thd = x
        th_ref, thd_ref = self.ref_theta(t)

        wx, wv = 20.0, 0.2
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

    def solve_ilqr(self, x0, U_init, t0):
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

            # terminal derivatives at absolute time t=t0+H*dt
            tH = t0 + self.H * self.dt
            Vx, Vxx = self.term_derivs_fd(X[-1], tH)

            for k in reversed(range(self.H)):
                t = t0 + k * self.dt
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
        self.t0 = 0.0


    def control(self, state, t):
        U_opt, _, _, _ = self.solve_ilqr(state, self.U_guess, self.t0)

        action = U_opt[0, 0] if U_opt.ndim == 2 else U_opt[0]
        action = float(np.clip(action, -1.0, 1.0))

        # ---- PRINT DEBUG INFO ----
        theta = state[2]
        theta_ref, _ = self.ref_theta(self.t0)
        print(
    f"\rstep: {t:4d} | theta: {np.rad2deg(theta):6.2f}° | "
    f"theta_ref: {np.rad2deg(theta_ref):6.2f}° | action: {action:6.3f}",
    end="",
    flush=True
)
        # --------------------------
        
        # Log data for Problem 5 analysis
        self.history.append({
            't': self.t0,
            'theta': state[2],
            'theta_dot': state[3],
            'theta_ref': theta_ref,
            'action': action
        })

        self.U_guess[:-1] = U_opt[1:]
        self.U_guess[-1] = 0

        self.t0 += self.dt
        return action

# ---- Env + controller ----
env = CartPoleEnv(
    render_mode=None,
    x_threshold=X_LIMIT,
    theta_threshold_radians=THETA_LIMIT,
    continuous_action=True,
    disturbance=DISTURBANCE
)

mpc = CartpoleMPC(H=H, wx=wx, wv=wv, wth=wth, wthd=wthd, wu=wu)
state, _ = env.reset()
mpc.reset()

# ---- Rollout storage ----
ts, xs, thetas, thetadots, thetas_ref, us = [], [], [], [], [], []

for t in range(T):
    u = mpc.control(state, t)
    state, _, terminated, truncated, _ = env.step(u)

    th_ref, _ = mpc.ref_theta(mpc.t0)

    ts.append(mpc.t0)
    xs.append(state[0])
    thetas.append(state[2])
    thetadots.append(state[3])
    thetas_ref.append(th_ref)
    us.append(u)

    if terminated or truncated:
        break

env.close()

ts = np.array(ts)
xs = np.array(xs)
thetas = np.array(thetas)
thetadots = np.array(thetadots)
thetas_ref = np.array(thetas_ref)
us = np.array(us)

# ---- Plots ----

# 1) theta vs reference
plt.figure()
plt.plot(ts, np.rad2deg(thetas), label="Actual θ")
plt.plot(ts, np.rad2deg(thetas_ref), "--", label="Target θ_ref")
plt.xlabel("Time (s)")
plt.ylabel("Pole angle (deg)")
plt.title("Problem 5: Sine wave tracking")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("mpc_sine_tracking.pdf", bbox_inches='tight', format='pdf')  # save figure for report
plt.show()

# 2) phase plot
plt.figure()
plt.plot(np.rad2deg(thetas), thetadots)
plt.scatter(np.rad2deg(thetas[0]), thetadots[0], marker="o", label="Start")
plt.xlabel("θ (deg)")
plt.ylabel("θ̇ (rad/s)")
plt.title("Problem 5: Phase-space (θ vs θ̇)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("mpc_phase_space.pdf", bbox_inches='tight', format='pdf')  # save figure for report
plt.show()

# 3) action over time
plt.figure()
plt.plot(ts, us)
plt.xlabel("Time (s)")
plt.ylabel("Action u")
plt.title("Control input over time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mpc_control_input.pdf", bbox_inches='tight', format='pdf')  # save figure for report
plt.show()

# 4) cart position over time (useful for report)
plt.figure()
plt.plot(ts, xs)
plt.xlabel("Time (s)")
plt.ylabel("Cart position x (m)")
plt.title("Cart position over time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mpc_cart_position.pdf", bbox_inches='tight', format='pdf')  # save figure for report
plt.show()
