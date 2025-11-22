import numpy as np
import os
import matplotlib.pyplot as plt
import params as par

def plot_test():
    base_dir = os.environ.get("BASE_DIR", "/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code")
    data = np.load("output_ps/results_seed42_test.npz")
    tunes_data = np.load(base_dir + "/integrator/tunes_phasespaces.npz")
    params_data = np.load(base_dir + "/actions_stuff/particle_data_phasespace.npz")

    times = params_data["times"]

    tunes = tunes_data["tunes"]

    x = data['x']
    y = data['y']

    n_particles = x.shape[0]
    n_subplots = 20
    n_indices = x.shape[1]
    idx_list = np.round(np.linspace(0, x.shape[1] - 1, 20)).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    tune_min, tune_max = np.min(tunes), np.max(tunes)

    l = idx_list
    idx_list_short = [l[0], l[4], l[8], l[12], l[16], l[19]]

    step = 10

    for i, idx in enumerate(idx_list_short):
        tune_vals = tunes[idx, :]
        n_points = x.shape[2]
        c_vals = np.repeat(tune_vals[::10], n_points)

        sc = axes[i].scatter(
            x[::step, idx, :].flatten(), 
            y[::step, idx, :].flatten(), 
            c=c_vals, 
            cmap='viridis', s=1, vmin=tune_min, vmax=tune_max
        )

        if idx == idx_list[-1]:
            mask = (tunes[idx, :] > (0.8 - 10e-2)) & (tunes[idx, :] < (0.8 + 10e-2))
            tunes_island = tunes[idx, :][mask]
            n_trapped = len(tunes_island)

            trapping_prob = n_trapped / n_particles * 100
            print(f"Trapping probability is: {trapping_prob:.2f}%")

        axes[i].set_xlabel("x", fontsize=16)
        axes[i].set_ylabel("y", fontsize=16)
        axes[i].set_aspect('equal', adjustable='box')
        axes[i].set_xlim(x_min-1, x_max+1)
        axes[i].set_ylim(y_min-1, y_max+1)
        percent = int(round(idx / (n_indices - 1) * 100))
        axes[i].set_title(rf"a = {par.a_lambda(times[idx]):.3f}, $\omega_m$ = {par.omega_lambda(times[idx])/par.omega_s:.2f}")

    fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='Tune')
    #plt.tight_layout()
    fig.suptitle(f"Trapping probability = {trapping_prob:.2f}%", fontsize=24)
    plt.show()

def load_results():
    results_dir = "/mnt/c/Users/emanu/\"OneDrive - Alma Mater Studiorum Università di Bologna\"/CERN_data/code/trapping/trapping_data/"
    trapping_probs = []
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".npz"):
                data = np.load(os.path.join(root, file))
                trapping_probs.append(float(data["trapping_prob"]))

if __name__ == "__main__":
    #plot_test()
    load_results()