import os
import importlib
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import ellipk

import functions as fn

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

def generate_grid(grid_lim, n_particles):
    X = np.linspace(-0.01, grid_lim, n_particles)
    Y = 0

    action, theta = fn.compute_action_angle_inverse(X, Y)

    kappa_squared_list = np.zeros(len(action))
    Omega_list = np.zeros(len(action))
    Q_list = np.zeros(len(action))
    P_list = np.zeros(len(action))

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

def generate_circle(radius, n_particles):
    X_list = np.empty(n_particles)
    Y_list = np.empty(n_particles)
    kappa_squared_list = np.empty(n_particles)
    Omega_list = np.empty(n_particles)
    Q_list = np.empty(n_particles)
    P_list = np.empty(n_particles)

    count = 0

    while count < n_particles:
        X = random.uniform(-radius, radius)
        Y = random.uniform(-radius, radius)

        if X**2 + Y**2 <= radius**2:
            X_list[count] = X
            Y_list[count] = Y
            count += 1

    action, theta = fn.compute_action_angle_inverse(X_list, Y_list)

    for i, act in enumerate(action):
        h_0 = fn.find_h0_numerical(act)
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
        if np.any(kappa_squared > 1):
            print("kappa2 > 1")
        kappa_squared_list[i] = kappa_squared
        Omega_list[i] = np.pi / 2 * (par.A / ellipk(kappa_squared))

    for i, (angle, freq, k2) in enumerate(zip(theta, Omega_list, kappa_squared_list)):
        Q, P = fn.compute_Q_P(angle, freq, k2)
        Q_list[i] = Q
        P_list[i] = P

    phi, delta = fn.compute_phi_delta(Q_list, P_list)
    phi = np.mod(phi, 2 * np.pi) 
    q_init = np.array(phi)
    p_init = np.array(delta)

    #q_init += 1
    #p_init += - np.min(p_init) + np.max(p_ps) + 0.001

    """phasespace_qp = np.load(f"./integrator/phasespace_qp_150_{machine}.npz")
    q_ps = phasespace_qp["q"]
    p_ps = phasespace_qp["p"]   

    plt.scatter(q_ps, p_ps, s=1)
    plt.scatter(q_init, p_init, s=1)
    plt.show()"""

    return q_init, p_init

def generate_gaussian(sigma, n_particles, x_center, x_min, x_max, y_min, y_max):
    X_list = []
    Y_list = []
    action_list = []
    theta_list = []

    sigma += 50/100 * sigma

    """name_dir = f"nu_{par.nu_m:.2f}"
    data_relax = np.load(f"./ipac_simulations/a_0.03/{name_dir}/relax_point_isl.npz")
    x_center = data_relax["x"]
    y_center = data_relax["y"]"""

    x_center = 0
    y_center = 0

    while len(X_list) < n_particles:
        X_try = np.random.normal(loc=x_center, scale=np.sqrt(sigma), size=n_particles)
        Y_try = np.random.normal(loc=y_center, scale=np.sqrt(sigma), size=n_particles)
        r=x_max
        mask = (X_try - x_center)**2 + (Y_try - y_center)**2 <= r**2
        X_try = X_try[mask]
        Y_try = Y_try[mask]

        action_try, theta_try = fn.compute_action_angle_inverse(X_try, Y_try)

        for X, Y, act, theta in zip(X_try, Y_try, action_try, theta_try):
            try:
                h_0 = fn.find_h0_numerical(act)
            except ValueError:
                continue
            X_list.append(X)
            Y_list.append(Y)
            action_list.append(act)
            theta_list.append(theta)
            if len(X_list) >= n_particles:
                break

    X_list = np.array(X_list[:n_particles])
    Y_list = np.array(Y_list[:n_particles])
    action = np.array(action_list[:n_particles])
    theta = np.array(theta_list[:n_particles])

    kappa_squared_list = np.empty(n_particles)
    Omega_list = np.empty(n_particles)
    Q_list = np.empty(n_particles)
    P_list = np.empty(n_particles)

    for i, act in enumerate(action):
        h_0 = fn.find_h0_numerical(act)
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
        kappa_squared_list[i] = kappa_squared
        Omega_list[i] = np.pi / 2 * (par.A / ellipk(kappa_squared))

    for i, (angle, freq, k2) in enumerate(zip(theta, Omega_list, kappa_squared_list)):
        Q, P = fn.compute_Q_P(angle, freq, k2)
        Q_list[i] = Q
        P_list[i] = P

    phi, delta = fn.compute_phi_delta(Q_list, P_list)
    phi = np.mod(phi, 2 * np.pi) 
    q_init = np.array(phi)
    p_init = np.array(delta)
    
    """"phasespace_qp = np.load(f"./integrator/phasespace_qp_150_{machine}.npz")
    q_ps = phasespace_qp["q"]
    p_ps = phasespace_qp["p"]

    plt.scatter(q_ps, p_ps, s=1)
    plt.scatter(q_init, p_init, s=1)
    plt.show()"""

    return q_init, p_init

def load_data_qp(filename):
    data = np.load(filename)
    q = data['q']
    p = data['p']

    q_init = np.array(q)
    p_init = np.array(p)

    phasespace_qp = np.load(f"./integrator/phasespace_qp_150_{machine}.npz")
    q_ps = phasespace_qp["q"]
    p_ps = phasespace_qp["p"]
    
    """q_init += 1
    p_init += - np.min(p_init) + np.max(p_ps) + 0.001

    plt.scatter(q_ps, p_ps, s=1)
    plt.scatter(q_init, p_init, s=1)
    plt.show()"""

    return q_init, p_init

def load_data_xy(filename):
    data = np.load(filename)
    x = data['x_out']
    y = data['y_out']

    x = x[0]
    y = y[0]

    action, theta = fn.compute_action_angle_inverse(x, y)

    kappa_squared_list = np.zeros(len(action))
    Omega_list = np.zeros(len(action))
    Q_list = np.zeros(len(action))
    P_list = np.zeros(len(action))

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

    """phasespace_qp = np.load(f"./integrator/phasespace_qp_150_{machine}.npz")
    q_ps = phasespace_qp["q"]
    p_ps = phasespace_qp["p"]

    plt.scatter(q_ps, p_ps, s=1)
    plt.scatter(q_init, p_init, s=1)
    plt.show()"""

    return q_init, p_init

# ---------------------------------------


if __name__ == "__main__":
    machine = os.environ.get("MACHINE").lower()
    
    init_type = sys.argv[1]
    grid_lim = float(sys.argv[2])
    sigma = float(sys.argv[3])
    n_particles = int(sys.argv[4])
    loaded_data = sys.argv[5] if len(sys.argv) > 5 else None   

    if loaded_data is not None:
        #q_init, p_init = load_data_xy(loaded_data)
        q_init, p_init = load_data_qp(loaded_data)

    elif init_type == "gaussian":
        if machine == "fcc":
            q_init, p_init = generate_gaussian(sigma, n_particles, 0, -grid_lim, grid_lim, -grid_lim, grid_lim)    # center FCC

        elif machine == "als":
            q_init, p_init = generate_gaussian(sigma, n_particles, 0, -grid_lim, grid_lim, -grid_lim, grid_lim)    # center ALS

    elif init_type == "circle":
        q_init, p_init = generate_circle(grid_lim, n_particles)
    elif init_type == "grid":
        q_init, p_init = generate_grid(grid_lim, n_particles)
    
    output_dir = "init_conditions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"init_qp_{n_particles}_{init_type}_{machine}.npz")
    np.savez(file_path, q=q_init, p=p_init)