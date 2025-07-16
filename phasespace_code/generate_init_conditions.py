import os
import sys
import numpy as np
from scipy.special import ellipk

import params as par
import functions as fn

def generate_grid(grid_lim, n_particles):
    X = np.linspace(-0.1, grid_lim, n_particles)
    Y = 0
    #X = 9.93
    #Y = -0.3
    
    action, theta = fn.compute_action_angle_inverse(X, Y)

    action = np.atleast_1d(action)
    theta = np.atleast_1d(theta)

    if isinstance(action, (list, np.ndarray)):
        kappa_squared_list = np.zeros(len(action))
        Omega_list = np.zeros(len(action))
        Q_list = np.zeros(len(action))
        P_list = np.zeros(len(action))
    else:
        kappa_squared_list = np.zeros(1)
        Omega_list = np.zeros(1)
        Q_list = np.zeros(1)
        P_list = np.zeros(1)

    for i, act in enumerate(action):
        h_0 = fn.find_h0_numerical(act)
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
        kappa_squared_list[i] = kappa_squared
        Omega_list[i] = np.pi / 2 * (par.A / ellipk(kappa_squared))

    for i, (angle, freq, k2) in enumerate(zip(theta, Omega_list, kappa_squared_list)):
        Q, P = fn.compute_Q_P(angle, freq, k2)
        Q_list[i] = Q
        P_list[i] = P

    Q = Q_list
    P = P_list 

    phi, delta = fn.compute_phi_delta(Q, P)
    phi = np.mod(phi, 2 * np.pi) 
    q_init = phi
    p_init = delta

    return q_init, p_init

def load_data(filename):
    data = np.load(filename)
    q = data['q']
    p = data['p']

    q_init = np.array(q)
    p_init = np.array(p)

    return q_init, p_init


# ---------------------------------------


if __name__ == "__main__":
    grid_lim = float(sys.argv[1])
    n_particles = int(sys.argv[2])
    loaded_data = sys.argv[3] if len(sys.argv) > 3 else None

    if loaded_data is not None:
        q_init, p_init = load_data(loaded_data)
    else:
        q_init, p_init = generate_grid(grid_lim, n_particles) 

    output_dir = "init_conditions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"qp.npz")
    np.savez(file_path, q=q_init, p=p_init)