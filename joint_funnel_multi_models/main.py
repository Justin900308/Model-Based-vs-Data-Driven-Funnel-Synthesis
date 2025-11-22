from util import const as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from util import Integrator, dynamics, linearization
import jax
import cvxpy as cp
import control
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from plotting import data_plotting, data_plotting_quadrotor

max_iter = 30
m = ct.m
n = ct.n
T = ct.T
dt = ct.dt
W_traj = ct.W_traj
K_traj = ct.K0_traj
Q_traj = ct.Q0_traj

########## Current dynamics
print("Dynamics:    ", ct.run)
if ct.run == "unicycle":
    from traj_upt import traj_gen
    from funnel_upt import funnel_gen
    from lipschitz import lipschitz_estimator
    from simulations import traj_sim
else:
    from traj_upt_quadrotor import traj_gen
    from funnel_upt_quadrotor import funnel_gen
    from lipschitz_quadrotor import lipschitz_estimator
    from simulations_quadrotor import traj_sim
########## select model modes
mode = 0

########### channel selections
C = ct.C_u
D = ct.D_u
E = ct.E_u
G = ct.G_u








## Main loop
for iter in range(max_iter):
    print("Main iteration", iter + 1)
    if iter == 0:
        #####################initialize traj and K0############################################
        ## initializing the hover traj
        x_0 = ct.x_0
        x_traj = np.zeros([T, n])
        x_traj[0] = x_0
        u_traj = np.zeros([T - 1, m])
        for t in range(T - 1):
            u_traj[t] = np.array([ct.mass * ct.g, 0, 0., 0])
            # u_traj[t] = np.zeros(4)
            x_traj[t + 1] = Integrator.RK4(dt, x_traj[t], u_traj[t], W_traj[t])
        ## traj gen
        [x_traj, u_traj, A_list, B_list, F_list] = traj_gen(x_traj, u_traj, Q_traj, K_traj, iter)
        ## get K0
        K_traj = np.zeros([T - 1, m, n])
        ## simulate trajs
        [x_traj_sim, u_traj_sim] = traj_sim(x_traj, u_traj, W_traj, K_traj, Q_traj, True, False, False)
        ## get true matrices
        [A_list_sim, B_list_sim, F_list_sim] = linearization.linearize(x_traj_sim[0], u_traj_sim[0], W_traj)
        A_list = np.array(A_list)
        A_list_sim = np.array(A_list_sim)
        ## compute Y_traj
        Y_traj = np.zeros((T, m, n))
        for t in range(T - 1):
            Y_traj[t] = K_traj[t] @ Q_traj[t]
        ## find local Lip const
        input_list = [A_list_sim, B_list_sim, F_list_sim, x_traj_sim[0], K_traj, Q_traj, u_traj_sim[0]]
        data_plotting_quadrotor(x_traj_sim, u_traj_sim, K_traj, Q_traj)
        gamma_traj = lipschitz_estimator(input_list, mode)
        ## plotting data
        # data_plotting(x_traj_sim, u_traj, K_traj, Q_traj)
        #####################end of initialization############################################
    ## funnel update
    [Q_traj, Y_traj, K_traj] = funnel_gen(x_traj, u_traj, A_list_sim, B_list_sim, F_list_sim, Q_traj, Y_traj, C, D, E,G,
                                          gamma_traj)

    ## plotting flags
    plt_flag = True
    if iter % 2 == 0 and iter != 0:
        plt_flag = True
    ## plotting data
    if plt_flag == True:
        data_plotting_quadrotor(x_traj_sim, u_traj_sim, K_traj, Q_traj)

    ## simulate trajs
    # [_, _] = traj_sim(x_traj, u_traj, W_traj, K_traj, Q_traj, True, True, plt_flag)
    # [_, _] = traj_sim(x_traj, u_traj, W_traj, K_traj, Q_traj, True, False, True)
    [x_traj_sim, u_traj_sim] = traj_sim(x_traj, u_traj, W_traj, K_traj, Q_traj, True, False, plt_flag)
    ## get true matrices
    [A_list_sim, B_list_sim, F_list_sim] = linearization.linearize(x_traj_sim[0], u_traj_sim[0], W_traj)
    ## traj update
    [x_traj, u_traj, A_list, B_list, F_list] = traj_gen(x_traj, u_traj, Q_traj, K_traj, iter)
    ## find local Lip const
    input_list = [A_list_sim, B_list_sim, F_list_sim, x_traj_sim[0], K_traj, Q_traj, u_traj_sim[0]]
    gamma_traj = lipschitz_estimator(input_list, mode)
