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
        x_traj_sim[0, t + 1] = Integrator.RK4(dt, x_traj_sim[0, t], u_t, np.zeros(nw))

    if is_plotting == True:
        plotting3d_fcn(x_traj_sim[0], Q_traj)

    return x_traj_sim, u_traj_sim
