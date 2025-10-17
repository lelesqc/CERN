import os
import sys
import numpy as np
import random
from scipy.special import ellipk
import matplotlib.pyplot as plt

import params_fcc
import functions as fn

par = params_fcc.Params()

def generate_grid(grid_lim, n_particles):
    #X = np.linspace(-0.01, grid_lim, n_particles)
    X = np.linspace(-grid_lim, grid_lim, n_particles)
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

    ps = np.load("./action_angle/phasespace_a0.050_nu0.80_extra_fcc.npz")
    x_ps = ps["x"]
    y_ps = ps["y"]

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
            print("yeah")
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

    return q_init, p_init

def generate_gaussian(sigma, n_particles, x_center=0, x_min=-2, x_max=2, y_min=-2, y_max=2):
    ps = np.load("./action_angle/phasespace_a0.050_nu0.80_extra_fcc.npz")
    x_ps = ps["x"]
    y_ps = ps["y"]

    X_list = []
    Y_list = []
    action_list = []
    theta_list = []

    while len(X_list) < n_particles:
        X_try = np.random.normal(loc=x_center, scale=sigma, size=n_particles)
        Y_try = np.random.normal(loc=0.0, scale=sigma, size=n_particles)
        mask = (X_try >= x_min) & (X_try <= x_max) & (Y_try >= y_min) & (Y_try <= y_max)
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

    #plt.scatter(x_ps, y_ps, s=1)
    #plt.scatter(X_list, Y_list, s=2)
    #plt.title(par.sigma)
    #plt.axis("square")
    #plt.show()

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

    return q_init, p_init

def load_data(filename):
    data = np.load(filename)
    q = data['q']
    p = data['p']

    ps_qp = np.load("./integrator/evolved_qp_phasespace.npz")
    q_ps = ps_qp["q"]
    p_ps = ps_qp["p"]

    p = p - np.min(p) + 0.027

    #plt.scatter(q_ps, p_ps)
    #plt.scatter(q, p, s=1)
    #plt.show()

    q_init = np.array(q)
    p_init = np.array(p)

    return q_init, p_init


# ---------------------------------------


if __name__ == "__main__":
    grid_lim = float(sys.argv[1])
    n_particles = int(sys.argv[2])
    loaded_data = sys.argv[3] if len(sys.argv) > 3 else None   

    var = par.sigma 

    if loaded_data is not None:
        q_init, p_init = load_data(loaded_data)
    else:
        #q_init, p_init = generate_grid(grid_lim, n_particles) 
        #q_init, p_init = generate_circle(grid_lim, n_particles)
        #q_init, p_init = generate_gaussian(grid_lim, n_particles, 10, 8, 10.5, -5, 5)    #ALS island
        #q_init, p_init = generate_gaussian(grid_lim, n_particles, 2.5, 2, 2.9, -1, 1)    #FCC island
        q_init, p_init = generate_gaussian(grid_lim, n_particles)    #FCC center

    output_dir = "init_conditions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"qp_{n_particles}.npz")
    np.savez(file_path, q=q_init, p=p_init)