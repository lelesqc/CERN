import os
import sys
import random
import numpy as np
import functions as fn
import matplotlib.pyplot as plt
from scipy.special import ellipk

import params_fcc as par

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

def generate_gaussian(sigma, n_particles, x_center, x_min, x_max, y_min, y_max):
    X_list = []
    Y_list = []
    action_list = []
    theta_list = []

    while len(X_list) < n_particles:
        X_try = np.random.normal(loc=x_center, scale=np.sqrt(sigma), size=n_particles)
        Y_try = np.random.normal(loc=0.0, scale=np.sqrt(sigma), size=n_particles)
        r=x_max
        mask = (X_try)**2 + (Y_try)**2 <= r**2        
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

    return q_init, p_init

def load_data_qp(filename):
    data = np.load(filename)
    q = data['q']
    p = data['p']

    q_init = np.array(q)
    p_init = np.array(p)

    return q_init, p_init


# ----------------------------------------


if __name__ == "__main__":
    radius = float(sys.argv[1])
    sigma = float(sys.argv[2])
    n_particles = int(sys.argv[3])
    loaded_data = sys.argv[4] if len(sys.argv) > 4 else None   

    if loaded_data is not None:
        #q_init, p_init = load_data_xy(loaded_data)
        q_init, p_init = load_data_qp(loaded_data)
    
    else:
        #q_init, p_init = generate_circle(radius, n_particles)
        q_init, p_init = generate_gaussian(sigma, n_particles, 0, -radius, radius, -radius, radius)

    output_dir = "init_conditions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "init_distribution.npz")
    np.savez(file_path, q=q_init, p=p_init)