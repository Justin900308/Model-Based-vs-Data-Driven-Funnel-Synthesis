import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from util import Integrator, dynamics
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from util import const as ct

T = ct.T
n = ct.n
m = ct.m
nw = ct.nw
C = ct.C_u
D = ct.D_u
gamma_min = 0.0001
gamma_default = 1
gamma_max = 10


# input_list = [A_list_sim, B_list_sim, F_list_sim, x_traj_sim, K_traj, Q_traj, u_traj_sim]
def lipschitz_estimator(input_list, mode):
    RK4_jit = jax.jit(Integrator.RK4, static_argnums=0)
    ## unpack data
    A_list_sim = input_list[0]
    B_list_sim = input_list[1]
    F_list_sim = input_list[2]
    x_traj_sim = input_list[3]
    K_traj = input_list[4]
    Q_traj = input_list[5]
    u_traj_sim = input_list[6]

    gamma_traj = np.zeros(T - 1)

    for t in range(T - 1):
        gamma_traj[t] = gamma_default
        ## default value if K is all zeros
        # if (K_traj[t] == np.zeros([m, n])).all():
        #     continue
        # Q_half_t = la.sqrtm(Q_traj[t])
        # [eta_samples_t, w_samples_t] = get_samples_t(Q_half_t)
        # A_t = A_list_sim[t]
        # B_t = B_list_sim[t]
        # F_t = F_list_sim[t]
        # K_t = K_traj[t]
        #
        # ## loop over samples at t
        # gamma_t = np.zeros(4096)
        # ## use jax to compute the propagation
        # x_sample_t = x_traj_sim[t] + eta_samples_t
        # u_sample_t = np.zeros([len(eta_samples_t[:, 0]), m])
        # for s in range(len(eta_samples_t[:, 0])):
        #     u_sample_t[s] = u_traj_sim[t] + K_t @ eta_samples_t[s]
        # x_sample_tp1 = jax.vmap(
        #     lambda x, u, w: RK4_jit(ct.dt, x, u, w),
        #     in_axes=(0, 0, 0)
        # )(x_sample_t, u_sample_t, w_samples_t)
        #
        # x_sample_tp1 = np.array(x_sample_tp1)
        # for s in range(len(eta_samples_t[:, 0])):
        #     eta_sample_t_s = eta_samples_t[s]
        #     w_sample_t_s = w_samples_t[s]
        #
        #     ## propagate the sample point
        #     x_sample_tp1_s = x_sample_tp1[s]
        #     # x_prop = Integrator.RK4(ct.dt, x_traj_sim[t], u_traj_sim[t], np.zeros(nw))
        #     eta_sample_tp1_s = x_sample_tp1_s - x_traj_sim[t + 1]
        #
        #     LHS = eta_sample_tp1_s - (A_t + B_t @ K_t) @ eta_sample_t_s - F_t @ w_sample_t_s
        #     mu = (C + D @ K_t) @ eta_sample_t_s
        #     gamma_t[s] = LA.norm(LHS) / LA.norm(mu)
        # gamma_traj[t] = np.max(gamma_t) / 0.5
        # print(gamma_traj[t])
        # if gamma_traj[t] < gamma_min:
        #     gamma_traj[t] = gamma_min
        # if gamma_traj[t] >= gamma_max:
        #     gamma_traj[t] = gamma_max
        print("Lip est progress: ", t / T * 100, "Lip_t: ", gamma_traj[t])
    return gamma_traj / 1


def get_samples_t(Q_half_t):
    N1 = 4  ## sample number
    N2 = N1 ** 2
    ## generate the sample points on the unit sphere
    ## nw = 3 n = 13, but we only sample the deviation from the position and omega
    ## 6 states to sample
    eta_samples_t = jnp.zeros([N1 ** 6, n])  ## state sample

    ## sample grid
    grid = jnp.linspace(-1.0, 1.0, N1)
    ## the 6D grid with each of axis having coord by grid
    eta_coords = jnp.stack(
        jnp.meshgrid(
            grid, grid, grid,
            grid, grid, grid,
            indexing="ij"
        ), axis=-1)
    ## reshape back to 4096 * 1
    eta_coords = eta_coords.reshape(-1, 6)
    idx = jnp.array([0, 1, 2, 10, 11, 12])
    eta_samples = eta_samples_t.at[:, idx].set(eta_coords) ## set the empty array with grid
    # Normalize to unit sphere
    norms = jnp.linalg.norm(eta_samples, axis=1, keepdims=True)
    norms = jnp.maximum(norms, 1e-8)  # avoid division by zero at the all-zero point
    eta_unit = eta_samples / norms
    eta_samples_t = (Q_half_t @ eta_unit.T).T



    ## 3 noises to sample
    w_samples_t = jnp.zeros([N2 ** 3, nw])  ## state sample

    ## sample grid
    grid = jnp.linspace(-1.0, 1.0, N2)
    ## the 6D grid with each of axis having coord by grid
    w_coords = jnp.stack(
        jnp.meshgrid(
            grid, grid, grid,
            indexing="ij"
        ), axis=-1)
    ## reshape back to 4096 * 1
    w_coords = w_coords.reshape(-1, 3)
    idx = jnp.array([0, 1, 2])
    w_samples = w_samples_t.at[:, idx].set(w_coords) ## set the empty array with grid
    # Normalize to unit sphere
    norms = jnp.linalg.norm(w_samples, axis=1, keepdims=True)
    norms = jnp.maximum(norms, 1e-8)  # avoid division by zero at the all-zero point
    w_samples_t = w_samples / norms
    return eta_samples_t, w_samples_t
