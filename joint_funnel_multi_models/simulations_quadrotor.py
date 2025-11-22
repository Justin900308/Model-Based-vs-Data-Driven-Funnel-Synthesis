import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm
from util import Integrator, dynamics
import jax
import cvxpy as cp
from util import const as ct
from plotting import plotting3d_fcn

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

T = ct.T
n = ct.n
m = ct.m
dt = ct.dt
obs = ct.obs
obs_r = ct.obs_r
num_obs = ct.num_obs
x_des = ct.x_des
nw = ct.nw
N = ct.N

rng = np.random.default_rng()
## noise for samples
W_traj_s = ct.W_traj_s


def traj_sim(x_traj, u_traj, W_traj, K_traj, Q_traj, is_multi, is_test, is_plotting):
    x_traj_sim = np.zeros([N + 1, T, n])
    u_traj_sim = np.zeros([N + 1, T - 1, m])
    ## the first one is the nominal traj
    x_traj_sim[0, 0] = x_traj[0]
    for t in range(T - 1):
        u_t = u_traj[t] + K_traj[t] @ (x_traj_sim[0, t] - x_traj[t])
        u_traj_sim[0, t] = u_t
        x_traj_sim[0, t + 1] = Integrator.RK4(dt, x_traj_sim[0, t], u_t, np.zeros(nw))
    if is_multi == True:
        ## generate the sample trajs IC
        x_0s = generate_sample_ini(x_traj, Q_traj)

        ## simulate all samples
        # for i in range(N):  #
        #     x_samples_i, u_samples_i = traj_prop(x_traj, x_0s[i], u_traj, K_traj, W_traj_s)
        print("propagating samples")
        x_traj_samples, u_traj_samples = jax.vmap(
            lambda x_0, w_s: traj_prop_jit(x_traj, x_0, u_traj, K_traj, w_s),
            in_axes=(0, 0)
        )(x_0s, W_traj_s)

        x_traj_sim[1:] = x_traj_samples
        u_traj_sim[1:] = u_traj_samples
    if is_plotting == True:
        plotting3d_fcn(x_traj_sim[0], x_traj_sim, Q_traj, is_multi)

    return x_traj_sim, u_traj_sim


def traj_prop(x_traj: jnp.array, x_0_i: jnp.array, u_traj: jnp.array, K_traj: jnp.array, W_traj_i: jnp.array):
    u_samples_i = jnp.zeros([T - 1, m])
    x_samples_i = jnp.zeros([T, n])
    x_samples_i = x_samples_i.at[0].set(x_0_i)
    for t in range(T - 1):
        u_t = u_traj[t] + K_traj[t] @ (x_samples_i[t] - x_traj[t])
        u_samples_i = u_samples_i.at[t].set(u_t)
        x_tp1 = Integrator.RK4(dt, x_samples_i[t], u_t, W_traj_i[t])  ## set 0 noise first
        x_samples_i = x_samples_i.at[t + 1].set(x_tp1)

    return x_samples_i, u_samples_i


traj_prop_jit = jax.jit(traj_prop)


def generate_sample_ini(x_traj, Q_traj):
    x_0s = np.zeros([N, n])
    ## deviation in the positions
    theta = np.linspace(0, 2 * np.pi, 10)
    phi = np.linspace(0, np.pi, 10)
    ## construct the unit sphere samples
    for i in range(10):  ## for z
        x_0s[10 * i:10 * (i + 1), 2] = np.cos(phi[i])
        for j in range(10):  ## for x and y
            x_0s[10 * i + j, 0:2] = np.array([np.cos(theta[j]), np.sin(theta[j])]) * np.sin(phi[i])

    ## project back to the ellipsoid
    Q_0_half = la.sqrtm(Q_traj[0])
    for i in range(N):
        x_0s[i] = Q_0_half @ x_0s[i] + x_traj[0]

    return x_0s
