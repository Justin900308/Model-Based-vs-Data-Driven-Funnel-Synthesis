import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
import jax.numpy as jnp

## select dynamcis
# run = "unicycle"
run = "quadrotor"

####### global variables
## simulation samples
if run == "unicycle":
    ## obstacles
    num_obs = 2
    obs = np.array([[4, 3], [8, 3]])
    obs_r = 1

    N = 10
    n = 3
    m = 2
    nw = 2
    n_p = 2
    n_q = 2
    tf = 8
    T = 61
    dt = tf / T
    time_traj = np.linspace(0, tf, T)
    gamma1 = 0.4
    ## initial and final states
    x_0 = np.array([1.0, 1.0, 0])
    x_des = np.array([10, 5.5, 0])
    ## initial funnel
    Q0_traj = np.zeros([T, n, n])
    K0_traj = np.zeros([T - 1, m, n])
    for t in range(T):
        Q0_traj[t] = np.diag([1, 1, 0.1]) * 0.1
    Q0_traj[-1] = np.diag([1, 1, 0.1]) * 0.1
    ## control constraints
    u1_max = 2
    u1_min = 0
    u2_max = 2
    ########### channel selections
    ## Unicycle
    C_u = np.array([[0, 0, 1],
                    [0, 0, 0]])
    D_u = np.array([[0, 0],
                    [1, 0]])
    E_u = np.array([[1, 0],
                    [0, 1],
                    [0, 0]])
    ## Unicycle 1
    G_u1 = np.zeros([2, 2])
## quadrotor
elif run == "quadrotor":
    ## initial and final states (pos, vel, quaternion, omega)
    x_0 = np.array([1.0, 1.0, -0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    x_des = np.array([5.0, 5.0, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    # x_des = np.array([2.0, 2.0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    ## obstacles
    num_obs = 2
    obs = np.array([[2, 2, -1.5], [4, 4, -3.5]])
    obs_r = 0.5
    N = 10
    n = 13
    m = 4
    nw = 3
    n_p = 10
    n_q = 8
    tf = 4
    T = 51
    dt = tf / T
    time_traj = np.linspace(0, tf, T)
    ## system parameters
    mass = 0.0293
    g = 9.81
    J = jnp.diag(jnp.array([1.8203e-5, 1.8186e-5, 3.4484e-5])) * 100
    ## control constraints
    tau_max = 0.00255
    T_max = mass * g * 1.5
    ########### channel selections
    I3 = np.eye(3)
    I4 = np.eye(4)

    ## block matrices
    O13 = np.zeros((1, 3))
    O14 = np.zeros((1, 4))
    O31 = np.zeros((3, 1))
    O33 = np.zeros((3, 3))
    O34 = np.zeros((3, 4))
    O41 = np.zeros((4, 1))
    O43 = np.zeros((4, 3))
    O44 = np.zeros((4, 4))

    E_u = np.block([
        [O33, O34, O33],
        [I3, O34, O33],
        [O43, I4, O43],
        [O33, O34, I3],
    ])

    C_u = np.block([
        [O43, O43, I4, O43],
        [O33, O33, O34, I3],
        [O13, O13, O14, O13],
    ])

    D_u = np.vstack([
        O44,
        O34,
        np.concatenate([np.array([[1.0]]), O13], axis=1),
    ])
    G_u = np.zeros([n_q, nw])


    ## initial funnel
    Q0_traj = np.zeros([T, n, n])
    for t in range(T):
        Q0_traj[t] = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0.01, 0.01, 0.01])
    K0_traj = np.zeros([T - 1, m, n])

x_traj = np.zeros([T, n])
x_traj[0] = x_0
## initial control
u_traj = np.zeros([T - 1, m])
## process noise for the nominal trajectory
W_traj = np.zeros([T - 1, nw])
rng = np.random.default_rng(123456789)
W_traj[:, 0] = rng.uniform(low=-1.0, high=1.0, size=(T - 1))
rng = np.random.default_rng(987654321)
W_traj[:, 1] = rng.uniform(low=-1.0, high=1.0, size=(T - 1))

## noise for samples
W_traj_s = rng.uniform(low=-1.0, high=1.0, size=(N, T - 1, nw))
