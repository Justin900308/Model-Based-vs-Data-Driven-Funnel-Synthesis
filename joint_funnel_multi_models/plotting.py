import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm

from joint_funnel_multi_models.util.const import W_traj_s
from util import Integrator, dynamics
import jax
import cvxpy as cp
from util import const as ct

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

T = ct.T
N = ct.N
time_traj = ct.time_traj
n = ct.n
W_traj = ct.W_traj
x_0 = ct.x_0
x_des = ct.x_des
obs_r = ct.obs_r
obs = ct.obs
num_obs = ct.num_obs


def data_plotting(x_traj_sim, u_traj_sim, K_traj, Q_traj):
    ## construct the funnel bounds
    x1_bounds = np.zeros([T, 2])
    x2_bounds = np.zeros([T, 2])
    for t in range(T):
        ## plotting the ellipsoid
        Q_t = Q_traj[t, 0:2, 0:2]
        Q_half = la.sqrtm(Q_t)
        # Eigen-decomposition (ascending order from eigh)
        vals, vecs = LA.eigh(Q_t)
        order = vals.argsort()[::-1]  # sort descending so index 0 is largest
        vals = vals[order]
        ## x1 upper and lower bounds
        x1_bounds[t, 0] = x_traj_sim[0, t, 0] + np.sqrt(vals[0])
        x1_bounds[t, 0] = x_traj_sim[0, t, 0] + np.sqrt(Q_t[0, 0])
        # x1_bound_2 = x_traj_sim[0, t, 0] + np.sqrt(Q_t[0,0])
        # x1_bound_3 = find_funnel_bound(x_traj_sim[0,t],u_traj_sim[t],Q_t,K_traj[t])
        # print("Method 1: ", x1_bounds[t,0], "Method 2: ", x1_bound_2, "Method 3:", x1_bound_3[0])
        x1_bounds[t, 1] = x_traj_sim[0, t, 0] - np.sqrt(vals[0])
        x1_bounds[t, 1] = x_traj_sim[0, t, 0] - np.sqrt(Q_t[0, 0])
        ## x2 upper and lower bounds
        x2_bounds[t, 0] = x_traj_sim[0, t, 1] + np.sqrt(vals[1])
        x2_bounds[t, 1] = x_traj_sim[0, t, 1] - np.sqrt(vals[1])
        x2_bounds[t, 0] = x_traj_sim[0, t, 1] + np.sqrt(Q_t[1, 1])
        x2_bounds[t, 1] = x_traj_sim[0, t, 1] - np.sqrt(Q_t[1, 1])
    plt.subplot(2, 1, 1)
    ## plot state
    for i in range(N + 1):
        plt.plot(time_traj[0:T - 1], x_traj_sim[i, 0:T - 1, 0])
    plt.plot(time_traj[0:T - 1], x1_bounds[0:T - 1, 0], "r", linewidth=2, label="x1 upper bound")
    plt.plot(time_traj[0:T - 1], x1_bounds[0:T - 1, 1], "r-.", linewidth=2, label="x1 lower bound")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("State (x1)")
    plt.subplot(2, 1, 2)
    for i in range(N + 1):
        plt.plot(time_traj[0:T - 1], x_traj_sim[i, 0:T - 1, 1])
    plt.plot(time_traj[0:T - 1], x2_bounds[0:T - 1, 0], "r", linewidth=2, label="x2 upper bound")
    plt.plot(time_traj[0:T - 1], x2_bounds[0:T - 1, 1], "r-.", linewidth=2, label="x2 lower bound")
    plt.xlabel("time")
    plt.ylabel("State (x2)")
    plt.legend()
    plt.show()

    ## plot the controls
    Q_u_traj = np.zeros([T - 1, ct.m, ct.m])
    u1_bounds = np.zeros([T - 1, 2])  ## upper and lower
    u2_bounds = np.zeros([T - 1, 2])
    for t in range(T - 1):
        Q_u_traj[t] = K_traj[t] @ Q_traj[t] @ K_traj[t].T
        u1_bounds[t, 0] = u_traj_sim[0, t, 0] + np.sqrt(Q_u_traj[t, 0, 0])  ## upper
        u1_bounds[t, 1] = u_traj_sim[0, t, 0] - np.sqrt(Q_u_traj[t, 0, 0])  ## lower

        u2_bounds[t, 0] = u_traj_sim[0, t, 1] + np.sqrt(Q_u_traj[t, 1, 1])  ## upper
        u2_bounds[t, 0] = np.maximum(u2_bounds[t, 0], np.max(u_traj_sim[:, t, 1]))
        u2_bounds[t, 1] = u_traj_sim[0, t, 1] - np.sqrt(Q_u_traj[t, 1, 1])  ## lower
        u2_bounds[t, 1] = np.minimum(u2_bounds[t, 1], np.min(u_traj_sim[:, t, 1]))
    plt.subplot(2, 1, 1)
    plt.plot(time_traj[0:T - 2], u1_bounds[0:T - 2, 0], "r", linewidth=2, label="u1 upper bound")
    plt.plot(time_traj[0:T - 2], u1_bounds[0:T - 2, 1], "r-.", linewidth=2, label="u1 lower bound")
    for i in range(N):
        plt.plot(time_traj[0:T - 2], u_traj_sim[i, 0:T - 2, 0])
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("Control (u1)")

    plt.subplot(2, 1, 2)
    plt.plot(time_traj[0:T - 2], u2_bounds[0:T - 2, 0], "r", linewidth=2, label="u2 upper bound")
    plt.plot(time_traj[0:T - 2], u2_bounds[0:T - 2, 1], "r-.", linewidth=2, label="u2 lower bound")
    for i in range(N):
        plt.plot(time_traj[0:T - 2], u_traj_sim[i, 0:T - 2, 1])
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("Control (u2)")

    plt.show()

    ##plot the process noise
    plt.subplot(2, 1, 1)
    for i in range(ct.N):
        plt.plot(time_traj[0:T - 1], W_traj_s[i, 0:T - 1, 0])
    plt.xlabel("time")
    plt.ylabel("Noise (w1)")
    plt.subplot(2, 1, 2)
    for i in range(ct.N):
        plt.plot(time_traj[0:T - 1], W_traj_s[i, 0:T - 1, 1])
    plt.xlabel("time")
    plt.ylabel("Noise (w2)")
    plt.show()
    return


def plot_ellipsoid_wireframe(ax, center, Q, color='g', n_pts=40):
    # Eigen-decomposition of Q (Q = R diag(vals) R^T)
    vals, vecs = LA.eigh(Q)  # vals ascending
    radii = np.sqrt(vals)  # semi-axis lengths

    # Parameter for circle
    theta = np.linspace(0, 2 * np.pi, n_pts)

    # Unit circles in three coordinate planes (local/principal frame)
    circle_xy = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=0)
    circle_xz = np.stack([np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=0)
    circle_yz = np.stack([np.zeros_like(theta), np.cos(theta), np.sin(theta)], axis=0)

    circles = [circle_xy, circle_xz, circle_yz]

    for C in circles:
        # scale by radii in principal coordinates
        C_scaled = radii[:, None] * C  # (3, n_pts)

        # rotate to world frame: x = center + vecs @ C_scaled
        pts = vecs @ C_scaled  # (3, n_pts)
        X = center[0] + pts[0, :]
        Y = center[1] + pts[1, :]
        Z = center[2] + pts[2, :]

        # your convention has -Z in plotting
        ax.plot(X, Y, -Z, color=color, linewidth=0.8)


def NED_2_plot_frame(v):
    return np.array([v[0], v[1], -v[2]])


def plotting3d_fcn(x_traj, Q_traj):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    ## create teh grid coordinates
    phi, theta = np.linspace(0, np.pi, 20), np.linspace(0, 2 * np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)

    for t in range(T):

        ## plotting the start and end points
        ax.plot(x_0[0], x_0[1], -x_0[2], "r.", markersize=10)
        ax.plot(x_des[0], x_des[1], -x_des[2], "r.", markersize=10)

        ## plotting the trajectory
        ax.plot(x_traj[:, 0], x_traj[:, 1], -x_traj[:, 2], "r-.", label="reference traj")

        ax.plot(x_traj[t, 0], x_traj[t, 1], -x_traj[t, 2], "r.", label="quadrotor")
        ax.legend()
        ## plotting the attitude
        q_t = x_traj[t, 6:10]
        x_body_t = NED_2_plot_frame(dynamics.quat_rotate(q_t, x_axis))
        x_body_t = x_body_t / LA.norm(x_body_t, 2)
        y_body_t = NED_2_plot_frame(dynamics.quat_rotate(q_t, y_axis))
        y_body_t = y_body_t / LA.norm(y_body_t, 2)
        z_body_t = NED_2_plot_frame(dynamics.quat_rotate(q_t, z_axis))
        z_body_t = z_body_t / LA.norm(z_body_t, 2)
        ax.plot(x_traj[t, 0] + np.array([0, x_body_t[0]]),
                x_traj[t, 1] + np.array([0, x_body_t[1]]),
                -x_traj[t, 2] + np.array([0, x_body_t[2]]), "r")
        ax.plot(x_traj[t, 0] + np.array([0, y_body_t[0]]),
                x_traj[t, 1] + np.array([0, y_body_t[1]]),
                -x_traj[t, 2] + np.array([0, y_body_t[2]]), "g")
        ax.plot(x_traj[t, 0] + np.array([0, z_body_t[0]]),
                x_traj[t, 1] + np.array([0, z_body_t[1]]),
                -x_traj[t, 2] + np.array([0, z_body_t[2]]), "b")

        ## plotting the ellipsoid
        Q_t = Q_traj[t, 0:3, 0:3]  ## the pos state
        ## create unit sphere
        x_ellip = np.sin(phi) * np.cos(theta)
        y_ellip = np.sin(phi) * np.sin(theta)
        z_ellip = np.cos(phi)

        ## project to the ellipsoid
        Q_half_t = la.sqrtm(Q_t)
        sphere_pts = np.stack([x_ellip, y_ellip, z_ellip], axis=-1)  # (..., 3)

        # Apply linear map Q_half_t: for each point v, map to Q_half_t @ v
        ellip_pts = sphere_pts @ Q_half_t.T

        x_ellip = ellip_pts[..., 0] + x_traj[t, 0]
        y_ellip = ellip_pts[..., 1] + x_traj[t, 1]
        z_ellip = ellip_pts[..., 2] + x_traj[t, 2]

        ax.plot_surface(x_ellip, y_ellip, -z_ellip, color='g', alpha=0.3, edgecolor='none')

        ## plotting the obstacles

        # Plot the sphere
        for i in range(num_obs):
            x = np.sin(phi) * np.cos(theta) * obs_r * 1 + obs[i, 0]
            y = np.sin(phi) * np.sin(theta) * obs_r * 1 + obs[i, 1]
            z = np.cos(phi) * obs_r * 1 + obs[i, 2]
            ax.plot_surface(x, y, -z, color='cyan', alpha=0.3, edgecolor='none')

        ax.set_xlim([0, 6])
        ax.set_ylim([0, 6])
        ax.set_zlim([0, 6])
        if t == T - 1:
            ## plot the entire funnel
            for tt in range(t):
                Q_t = Q_traj[tt, 0:3, 0:3]
                center_t = x_traj[tt, 0:3]
                plot_ellipsoid_wireframe(ax, center_t, Q_t, color='g', n_pts=30)
            ax.view_init(elev=0, azim=0, roll=0)
            plt.show()
        else:
            plt.pause(0.01)
            ax.clear()
