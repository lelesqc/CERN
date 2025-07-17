import sys
import numpy as np
import matplotlib.pyplot as plt

import params as par

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

if __name__ == "__main__":
    plot_both_with_tune()