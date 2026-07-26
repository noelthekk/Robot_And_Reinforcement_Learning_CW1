import numpy as np
import matplotlib.pyplot as plt


def plot_policy_value(policy, V, N_bins, x_vals, theta_vals):
    mid = N_bins // 2
    v_slice = V[:, mid, :, mid].T
    p_slice = policy[:, mid, :, mid].T

    print("unique values in p_slice:", np.unique(p_slice))
    print("dtype:", p_slice.dtype)


    dx = x_vals[1] - x_vals[0]
    dt = np.rad2deg(theta_vals[1] - theta_vals[0])

    extent = [
        x_vals[0] - dx/2, x_vals[-1] + dx/2, 
        np.rad2deg(theta_vals[0]) - dt/2, np.rad2deg(theta_vals[-1]) + dt/2
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(v_slice, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    # im2 = ax2.imshow(p_slice, origin='lower', extent=extent, aspect='auto', cmap='managua')
    im2 = ax2.imshow(p_slice, origin='lower', extent=extent, aspect='auto',cmap='managua', vmin=0, vmax=1)

    fig.colorbar(im1, ax=ax1,)
    # fig.colorbar(im2, ax=ax2, ticks=[0, 1])
    cbar = fig.colorbar(im2, ax=ax2, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", "0.5 (tie)", "1"])
    ax1.set_title(r"Value Function $V(x, \dot{x}=0, \theta, \dot{\theta}=0)$")
    ax2.set_title(r"Policy $\pi(s)$")    

    x_edges = np.linspace(x_vals[0] - dx/2, x_vals[-1] + dx/2, N_bins + 1)
    t_edges = np.linspace(np.rad2deg(theta_vals[0]) - dt/2, np.rad2deg(theta_vals[-1]) + dt/2, N_bins + 1)

    for ax in [ax1, ax2]:
        ax.set_xticks(x_edges, minor=True)
        ax.set_yticks(t_edges, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.6)
        ax.axhline(0, color='white', linewidth=1.5, alpha=0.7) # Horizontal zero (Theta=0)
        ax.axvline(0, color='white', linewidth=1.5, alpha=0.7) # Vertical zero (X=0)
        ax.set_axisbelow(False)
        ax.set_xlabel('Cart Position (m)')
        ax.set_ylabel('Pole Angle (deg)')

    plt.tight_layout()
    plt.show()

def plot_returns(returns, timesteps):
    plt.plot(timesteps, returns)
    plt.xlabel('Timesteps (M)')
    plt.ylabel('Return')
    plt.title('Returns')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('returns.png', dpi=300, bbox_inches='tight', pad_inches=0.05)


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


def plot_3d_slice(data,
                  x_vals,
                  x_dot_vals,
                  theta_vals,
                  theta_dot_vals,
                  fixed_var="x",
                  fixed_value=0.0,
                  is_policy=False):
    """
    3D scatter of a 4D array (V or policy).

    Default axes: x_dot, theta, theta_dot
    Fixed variable default: x = 0
    """

    grids = {
        "x": x_vals,
        "xdot": x_dot_vals,
        "theta": theta_vals,
        "thetadot": theta_dot_vals
    }

    # find index closest to fixed_value
    fixed_idx = np.argmin(np.abs(grids[fixed_var] - fixed_value))

    # build slice
    if fixed_var == "x":
        slice_data = data[fixed_idx, :, :, :]
    elif fixed_var == "xdot":
        slice_data = data[:, fixed_idx, :, :]
    elif fixed_var == "theta":
        slice_data = data[:, :, fixed_idx, :]
    elif fixed_var == "thetadot":
        slice_data = data[:, :, :, fixed_idx]
    else:
        raise ValueError("Invalid fixed_var")

    # default axes = xdot, theta, thetadot
    X, Y, Z = np.meshgrid(
        x_dot_vals,
        theta_vals,
        theta_dot_vals,
        indexing="ij"
    )

    values = slice_data.flatten()
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(Xf, Yf, Zf,
                    c=values,
                    cmap="managua" if is_policy else "viridis",
                    s=8)

    ax.set_xlabel("x_dot")
    ax.set_ylabel("theta")
    ax.set_zlabel("theta_dot")

    fig.colorbar(sc, shrink=0.6)
    plt.tight_layout()
    plt.show()

import numpy as np

def plot_3d_slice_interactive(data,
                              x_vals, x_dot_vals, theta_vals, theta_dot_vals,
                              fixed_var="x", fixed_value=0.0,
                              is_policy=False,
                              sample=20000):
    import plotly.graph_objects as go

    grids = {"x": x_vals, "xdot": x_dot_vals, "theta": theta_vals, "thetadot": theta_dot_vals}
    fixed_idx = np.argmin(np.abs(grids[fixed_var] - fixed_value))

    if fixed_var == "x":
        slice_data = data[fixed_idx, :, :, :]
    elif fixed_var == "xdot":
        slice_data = data[:, fixed_idx, :, :]
    elif fixed_var == "theta":
        slice_data = data[:, :, fixed_idx, :]
    elif fixed_var == "thetadot":
        slice_data = data[:, :, :, fixed_idx]
    else:
        raise ValueError("fixed_var must be one of: x, xdot, theta, thetadot")

    X, Y, Z = np.meshgrid(x_dot_vals, theta_vals, theta_dot_vals, indexing="ij")

    vals = slice_data.reshape(-1)
    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    Zf = Z.reshape(-1)

    # optional subsample for speed
    if sample is not None and len(vals) > sample:
        idx = np.random.choice(len(vals), size=sample, replace=False)
        Xf, Yf, Zf, vals = Xf[idx], Yf[idx], Zf[idx], vals[idx]

    fig = go.Figure(data=go.Scatter3d(
        x=Xf, y=Yf, z=Zf,
        mode='markers',
        marker=dict(
            size=2.5,
            color=vals,
            colorscale='Viridis' if not is_policy else 'Turbo',
            opacity=0.8,
            colorbar=dict(title="policy" if is_policy else "V")
        )
    ))

    fig.update_layout(
        width=900, height=650,
        scene=dict(
            xaxis_title="x_dot",
            yaxis_title="theta",
            zaxis_title="theta_dot"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"3D slice (fixed {fixed_var}≈{grids[fixed_var][fixed_idx]:.4f})"
    )

    fig.show()


