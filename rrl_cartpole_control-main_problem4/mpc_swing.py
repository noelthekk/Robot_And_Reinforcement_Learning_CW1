from matplotlib.projections import HammerAxes
import matplotlib.pyplot as plt
import numpy as np
from my_cartpole_env import CartPoleEnv
from gymnasium.wrappers import TimeLimit, RecordVideo
import os
from hyperparameters import *
from common import *
from plotting import plot_returns


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


    def control(self, state):
        U_opt, _, _, _ = self.solve_ilqr(state, self.U_guess, self.t0)

        action = U_opt[0, 0] if U_opt.ndim == 2 else U_opt[0]
        action = float(np.clip(action, -1.0, 1.0))

        # ---- PRINT DEBUG INFO ----
        theta = state[2]
        theta_ref, _ = self.ref_theta(self.t0)
        print(f"theta: {np.rad2deg(theta):6.2f}°, "
            f"theta_ref: {np.rad2deg(theta_ref):6.2f}°, "
            f"action: {action:5.2f}")
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




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Base Configuration for Problem 5
    FIXED_H = 25 # Increased to help the controller "see" the sine wave peaks
    num_evals = 200   
    max_steps = 500    
    DISTURBANCE = 0
    base_params = {'wx': 0.2, 'wv': 0.1, 'wth': 50.0, 'wthd': 1.0, 'wu': 0.001}
    
    # Updated ranges to test more "aggressive" tracking
    variations = {
        # 'wx': [0.1, 0.2, 0.5]
        'wv': [0.01, 0.1, 0.5] 
        # 'wth': [30.0, 50.0, 70.0], 
        # 'wthd': [0.5, 1.0, 2.0], 
        # 'wu': [0.0005, 0.001, 0.005] 
    }

    sensitivity_data = {param: {'values': [], 'rates': []} for param in variations}

    for param_name, test_values in variations.items():
        for val in test_values:
            current_params = base_params.copy()
            current_params[param_name] = val
            
            print(f"Testing {param_name} = {val} over {num_evals} episodes...")
            mpc = CartpoleMPC(H=FIXED_H, **current_params)
            
            # Setup play_env
            play_env = CartPoleEnv(render_mode="rgb_array", 
                                  x_threshold=X_LIMIT, 
                                  theta_threshold_radians=THETA_LIMIT, 
                                  continuous_action=True, 
                                  disturbance=DISTURBANCE)
            play_env = TimeLimit(play_env, max_episode_steps=max_steps)
            
            video_prefix = f"H50_{param_name}_{val}"
            play_env = RecordVideo(play_env, 
                                   video_folder="videos", 
                                   name_prefix=video_prefix, 
                                   episode_trigger=lambda ep: ep == 0) # Just record 1 video to save time

            terminated_count = 0
            for ep in range(num_evals):
                state, _ = play_env.reset()
                mpc.reset()
                
                for t in range(max_steps):
                    action = mpc.control(state)
                    state, _, terminated, truncated, _ = play_env.step(action)
                    if terminated:
                        terminated_count += 1
                        break
                    if truncated:
                        break
            
            rate = terminated_count / num_evals
            sensitivity_data[param_name]['values'].append(val)
            sensitivity_data[param_name]['rates'].append(rate)
            play_env.close()

    # --- 2. Generate and Save PDF Summary ---
    # Using squeeze=False ensures 'axes' is an array even with 1 variation
    fig, axes = plt.subplots(1, len(variations), figsize=(6 * len(variations), 5), squeeze=False, sharey=True)
    
    # Flatten allows us to use axes[i] regardless of the number of variations
    axes_flat = axes.flatten() 

    for i, (param, data) in enumerate(sensitivity_data.items()):
        axes_flat[i].plot(data['values'], data['rates'], marker='o', color='crimson', linestyle='--')
        axes_flat[i].set_title(f'Failure Rate vs {param}')
        axes_flat[i].set_xlabel('Weight Value')
        axes_flat[i].set_ylim([-0.05, 1.05])
        
        # Only label the y-axis for the first plot in the row
        if i == 0: 
            axes_flat[i].set_ylabel('Termination Rate')
        
        axes_flat[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Updated filename to reflect the specific test
    pdf_filename = 'weight_sensitivity_wx_only.pdf'
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    print(f"\nAnalysis complete. Plot saved to {pdf_filename}", flush=True)
    plt.show()
