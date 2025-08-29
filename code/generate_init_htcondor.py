import numpy as np
import random
from scipy.special import ellipk
import functions as fn
import params as par
import yaml
import os
from tqdm import tqdm

def generate_init(radius, n_particles, seed):
    """
    Generate circular initial conditions in (q, p) space starting from (X, Y) coordinates,
    using a specified random seed for reproducibility.

    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

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


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    radius = float(params["radius"])
    n_particles = int(params["particles"])
    seed = int(params["seed"])

    output_dir = "init_htcondor"
    os.makedirs(output_dir, exist_ok=True)

    q_init, p_init = generate_init(radius, n_particles, seed)
    outname = f"init_distribution_{seed}.npz"
    file_path = os.path.join(output_dir, outname)
    np.savez(file_path, q=q_init, p=p_init, seed=seed)