import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import params as par
import functions as fn

def plot_both():
    phase_file = f"action_angle/phasespace_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz"
    evol_file = f"../code/action_angle/last_a0.025-0.050_nu0.90-0.80.npz"

    data_phase = np.load(phase_file)
    data_evol = np.load(evol_file)

    plt.figure(figsize=(7,7))
    plt.scatter(data_phase['x'], data_phase['y'], s=10, label="Phase space", alpha=0.7)
    plt.scatter(data_evol['x'], data_evol['y'], s=10, label="Evolution", alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_both_with_tune():
    data_fft = np.load("tune_analysis/fft_results.npz")
    phase_file = np.load(f"action_angle/phasespace_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
    evolved_for_fft = np.load(f"action_angle/tune_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

    tunes = data_fft['tunes_list']
    tunes_list = [[] for _ in range(len(phase_file['x']))]

    for i in range(len(tunes)):
        for _ in range(len(phase_file['x'][0])):
            tunes_list[i].append(tunes[i])     

    x_last = evolved_for_fft['x'][-1, :]
    y_last = evolved_for_fft['y'][-1, :]

    plt.figure(figsize=(7,7))
        
    sc = plt.scatter(phase_file['x'], phase_file['y'], c=tunes_list, s=10, label="Phase space", alpha=1.0)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis('equal')
    plt.colorbar(sc, label="Tune")
    plt.tight_layout()
    plt.show()

def plot_actions():
    data_init = np.load("actions_analysis/init_actions_10000.npz")
    data_final = np.load("actions_analysis/final_actions_10000.npz")
    final_actions = data_final['final_actions']
    init_actions = data_init['init_actions']
    tunes = np.load("tune_analysis/tunes_results.npz")['tunes_list']

    print(tunes.shape)


    plt.figure(figsize=(7,7))
    sc = plt.scatter(init_actions, final_actions, c=tunes, s=1, cmap='plasma', alpha=0.7)
    plt.xlabel("initial actions")
    plt.ylabel("final actions")
    plt.colorbar(sc, label="Tune")
    plt.axis('square')
    plt.title("Initial vs Final Actions with Tune, 10'000 particles")
    plt.tight_layout()
    plt.show()

def plot_init_distr_coloured():
    data_init = np.load("../code/init_conditions/init_distribution.npz")
    data_tune = np.load("tune_analysis/tunes_results.npz")

    tunes = data_tune['tunes_list']
    q = data_init['q']
    p = data_init['p']

    print(q.shape)

    n_particles = q.shape[0]

    actions_list = np.zeros(n_particles)

    x = np.zeros(n_particles)
    y = np.zeros(n_particles)
    
    for i in tqdm(range(n_particles)):
        h_0 = fn.H0_for_action_angle(q[i], p[i])
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

        if 0 < kappa_squared < 1:
            Q = (q[i] + np.pi) / par.lambd
            P = par.lambd * p[i]

            action, theta = fn.compute_action_angle(kappa_squared, P)
            actions_list[i] = action 

            x[i] = np.sqrt(2 * action) * np.cos(theta)
            y[i] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q[i]-np.pi)

    x = np.array(x)
    y = np.array(y)

    plt.figure(figsize=(7,7))
    sc = plt.scatter(x, y, c=tunes, cmap='plasma', s=2.5, alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(sc, label="Tune")
    plt.axis('square')
    plt.title("Initial Distribution Coloured by Tune")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    #plot_both_with_tune()
    #plot_actions()
    plot_init_distr_coloured()