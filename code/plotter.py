import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import params as par

base_dir = os.environ["BASE_DIR"]

def plot(poincare_mode, n_particles, n_to_plot):
    a_start = f"{par.a_lambda(par.T_percent):.3f}"
    omega_start = f"{par.omega_lambda(par.T_percent):.2f}"
    a_end = f"{par.a_lambda(par.T_tot):.3f}"
    omega_end = f"{par.omega_lambda(par.T_tot):.2f}"

    nu_f = par.omega_lambda(par.T_tot)/par.omega_s

    str_title = f"a{a_start}-{a_end}_nu{float(omega_start)/par.omega_s:.2f}-{float(omega_end)/par.omega_s:.2f}_{n_particles}"

    data = np.load(base_dir + f"/action_angle/{poincare_mode}_{str_title}.npz")

    x = data['x']
    y = data['y']

    if poincare_mode in ["first", "last"]:
        tunes_data = np.load(base_dir + "/integrator/evolved_qp_last_10000.npz")
        tunes = tunes_data["tunes"]

        mask = (tunes > (nu_f - 10e-3)) & (tunes < (nu_f + 10e-3))

        tunes_island = tunes[mask]
        n_trapped = len(tunes_island)

        x_in, y_in = x[mask], y[mask]
        x_out, y_out = x[~mask], y[~mask]

        stats = {
            "mean_x_in": np.mean(x_in),
            "mean_y_in": np.mean(y_in),
            "var_x_in": np.var(x_in),
            "var_y_in": np.var(y_in),

            "mean_x_out": np.mean(x_out),
            "mean_y_out": np.mean(y_out),
            "var_x_out": np.var(x_out),
            "var_y_out": np.var(y_out),

            "trapping_prob": n_trapped / x.shape[0] * 100
        }

        print(f"Trapping probability: {stats["trapping_prob"]:.1f} %")

        results_dir = base_dir + "/trapping/trapping_data"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        #np.savez(results_dir + f"/results_{str_title}.npz", **stats)

        plt.figure(figsize=(7,7))
        sc = plt.scatter(x, y, c=tunes, cmap="viridis", s=4, alpha=1.0)
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=16)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.tick_params(labelsize=18)
        plt.colorbar(sc, label="Tune")
        plt.title(f"Trapping probability: {stats["trapping_prob"]:.1f}%")
        plt.tight_layout()

        output_dir = base_dir + "/trapping" + "/trapping_pics"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        #plt.savefig(output_dir + "/" + str_title + ".png")

        plt.show()

    elif poincare_mode == "all":
        n_sections = len(x) // n_particles
        idx_to_plot = np.linspace(0, n_sections-1, n_to_plot, dtype=int)

        _, axes = plt.subplots(1, n_to_plot, figsize=(4*n_to_plot, 4), sharex=True, sharey=True)
        if n_to_plot == 1:
            axes = [axes]
        for ax, i in zip(axes, idx_to_plot):
            ax.scatter(x[i*n_particles:(i+1)*n_particles], y[i*n_particles:(i+1)*n_particles], s=2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y") 

        plt.suptitle(f"Poincaré sections for {poincare_mode} mode", fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust top to make room for the suptitle
        plt.show()

    elif poincare_mode == "none":
        plt.figure(figsize=(7,7))
        plt.scatter(x, y, s=3, label=r"Final distribution", alpha=1.0)
        print(x.shape)
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=16)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.legend(fontsize=20)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()

# -----------------------------------------


if __name__ == "__main__":
    poincare_mode = str(sys.argv[1])
    n_particles = int(sys.argv[2])
    n_to_plot = int(sys.argv[3])
    
    plot(poincare_mode, n_particles, n_to_plot)
