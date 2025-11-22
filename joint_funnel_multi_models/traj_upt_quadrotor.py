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

## obstacles
num_obs = ct.num_obs
obs = ct.obs
obs_r = ct.obs_r

## global variables
n = ct.n
m = ct.m
dt = ct.dt
T = ct.T
x_0 = ct.x_0
x_des = ct.x_des
x_traj = np.zeros([T, n])
x_traj[0] = x_0
u_traj = np.zeros([T - 1, m])
W_traj = ct.W_traj
mass = ct.mass
K_traj = ct.K0_traj
Q_traj = ct.Q0_traj

## control constraints
tau_max = ct.tau_max


## simulate the traj and check linearization
# for t in range(T - 1):
#     u_traj[t] = np.array([mass * ct.g, 0, 0., 0])
#     # u_traj[t] = np.zeros(4)
#     x_traj[t + 1] = Integrator.RK4(dt, x_traj[t], u_traj[t], W_traj[t])


# plotting3d_fcn(x_traj, Q_traj)


def linearization(integrator, dt, x_t, u_t, W_t) -> tuple:
    f = lambda x, u, W: integrator(dt, x, u, W)
    # Compute the Jacobian of f(x, u) with respect to x (A matrix)
    A_k = jax.jacobian(lambda x: f(x, u_t, W_t))(x_t)
    # Compute the Jacobian of f(x, u) with respect to u (B matrix)
    B_k = jax.jacobian(lambda u: f(x_t, u, W_t))(u_t)
    # Compute the Jacobian of f(x, u) with respect to W (F matrix)
    W_k = jax.jacobian(lambda W: f(x_t, u_t, W))(W_t)
    return A_k, B_k, W_k


linearization_jit = jax.jit(linearization, static_argnums=0)


def cost_subproblem_fun(x_traj, u_traj, x_des, d, w, v, s):
    ## terminal cost
    f0 = 10000 * cp.norm(x_traj[T - 1] + d[T - 1] - x_des, 2)

    ## running cost
    for t in range(T - 1):
        ## for objective
        # f0 += cp.norm(x_traj[t] + d[t] - x_des, 2) * 0.4 * (t)
        # f0 += cp.norm(x_traj[t,2] + d[t,2],2)
        f0 += 20*cp.norm(u_traj[t,0] + w[t,0], 2) ** 2
        ## for constraints
        f0 += 1000 * cp.norm(v[t], 1)
        # ## for obs
        for j in range(num_obs):
            f0 += 1000 * cp.abs(s[t, j])
        ## regularization
        f0 += 50 * cp.norm(w[t], 2)

    return f0


def solve_subproblem(A_list, B_list, trajs, x_des) -> tuple:
    x_traj = trajs[0]
    u_traj = trajs[1]
    W_traj = trajs[2]
    Q_traj = trajs[3]
    K_traj = trajs[4]
    d = cp.Variable([T, n])
    w = cp.Variable([T - 1, m])
    v = cp.Variable([T, n])
    s = cp.Variable([T, num_obs])
    ## starting constraints
    constraints = [d[0] == np.zeros(n)]
    ## terminal constraints
    # constraints.append(d[-1]+x_traj[-1] == x_des)

    for t in range(T - 1):
        x_t = x_traj[t]
        x_tp1 = x_traj[t + 1]
        u_t = u_traj[t]
        w_t = w[t]
        W_t = W_traj[t]
        d_t = d[t]
        d_tp1 = d[t + 1]
        v_t = v[t]
        s_t = s[t]
        A_t = A_list[t]
        B_t = B_list[t]
        Q_t = Q_traj[t]
        Q_t_half = la.sqrtm(Q_t)
        K_t = K_traj[t]
        f_t = Integrator.RK4(dt, x_t, u_t, W_t)
        constraints.append(x_tp1 + d_tp1 == A_t @ d_t + B_t @ w_t + f_t + 1 * np.diag(np.ones(n)) @ v_t)

        #### input constraints
        Q_u_t = K_t @ Q_t @ K_t.T  ## control funnel
        Q_u_t_half = la.sqrtm(Q_u_t)
        ## max thrust
        a_t_u1_max = np.array([1, 0, 0, 0])
        LHS_u1_max = cp.abs(u_t[0] + w_t[0]) + LA.norm(Q_u_t_half @ a_t_u1_max, 2)
        constraints.append(LHS_u1_max <= ct.T_max)
        ## max torque
        a_t_tau_max = np.zeros(4)
        for tau_i in range(3):
            a_t_tau_max[tau_i] = 1
            LHS_tau_i = cp.abs(u_t[tau_i + 1] + w_t[tau_i + 1])
            # LHS_tau_i += LA.norm(Q_u_t_half @ a_t_tau_max, 2)
            constraints.append(LHS_tau_i <= tau_max)

        ## ground constraint
        a_t_gnd = np.array([0, 0, 1])
        a_t_gnd = np.hstack((a_t_gnd, np.zeros(10)))
        LHS_gnd = x_t[2] + d_t[2] + LA.norm(Q_t_half @ a_t_gnd, 2) * 1
        constraints.append(LHS_gnd <= 0)

        ## obs constraints
        for j in range(num_obs):
            obs_j = obs[j]
            h_j = obs_r ** 2 - LA.norm(x_t[0:3] - obs_j[0:3], 2) ** 2
            a = - 2 * (x_t[0:3] - obs_j[0:3])
            LHS = h_j + a @ d_t[0:3]  ## obs constraints
            Q_t_root = sqrtm(Q_t[0:3, 0:3])
            LHS += LA.norm(Q_t_root @ a, 2)
            constraints.append(LHS <= s_t[j])
            # constraints.append(h_j - 2 * (x_t[0:2] - obs_j) @ d_t[0:2] <= s_t[j])
            constraints.append(s_t[j] >= 0)
    #
    # f0 = cost_subproblem_fun(x_traj, u_traj, x_des, d, w, v, s)
    f0 = cost_subproblem_fun(x_traj, u_traj, x_des, d, w, v, s)
    problem = cp.Problem(cp.Minimize(f0), constraints)
    problem.solve(solver=cp.CLARABEL)
    d_traj = d.value
    w_traj = w.value
    v_val = v.value
    true_cost = 0
    subproblem_cost = problem.value

    return d_traj, w_traj, true_cost, subproblem_cost


def traj_gen(x_traj, u_traj, Q_traj, K_traj, main_iter) -> tuple:
    if main_iter == 0:
        max_iter = 20
    else:
        max_iter = 20
    W_traj = np.zeros([T - 1, ct.nw])
    subproblem_cost_old = 0
    for iter in range(max_iter):
        [A_list, B_list, F_list] = jax.vmap(
            lambda x, u, W: linearization_jit(Integrator.RK4, dt, x, u, W),
            in_axes=(0, 0, 0)
        )(x_traj[0:T - 1, :], u_traj, W_traj)

        trajs = [x_traj, u_traj, W_traj, Q_traj, K_traj]
        [d_traj, w_traj, true_cost, subproblem_cost] = solve_subproblem(A_list, B_list, trajs, x_des)
        print("Traj upt iteration:  ", iter + 1, "subproblem cost:    ", subproblem_cost)
        ## update
        if d_traj.any() != None:
            x_traj += d_traj
            u_traj += w_traj
        print("cost diff:   ", np.abs(subproblem_cost_old - subproblem_cost))
        if np.abs(subproblem_cost_old - subproblem_cost) <= 0.01 or iter > 30:
            print("Sol converged")
            break

        # if np.abs(subproblem_cost_old - subproblem_cost) <= 0.01 or iter > 25:
        #     plotting3d_fcn(x_traj, Q_traj)
        subproblem_cost_old = subproblem_cost
        ## plotting
        # if iter % 5 == 0:
        #     plotting_fcn(x_traj, u_traj)
    # plotting3d_fcn(x_traj, Q_traj)
    return x_traj, u_traj, A_list, B_list, F_list

#
# [x_traj, u_traj, A_list, B_list] = traj_gen(x_traj, u_traj, Q_traj, K_traj, 0)
