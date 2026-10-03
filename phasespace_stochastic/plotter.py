import os
import importlib
import sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from pathlib import Path

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

machine = os.environ.get("MACHINE").lower()

def plot(mode, n_particles, config):
    if mode == "evolution":
        data = np.load(f"action_angle/evolution_{n_particles}_a{par.a:.7f}_nu{par.omega_m/par.omega_s:.5f}_{machine}_{config}.npz")
        phasespace = np.load(f"action_angle/phasespace_75_a{par.a:.7f}_nu{par.omega_m/par.omega_s:.5f}_{machine}_{config}.npz")

        #data_qp = np.load(f"integrator/{mode}_qp_{n_particles}_{machine}.npz")
        #phasespace_qp = np.load(f"./integrator/phasespace_qp_150_{machine}.npz")

        #q = data_qp["q"]
        #p = data_qp["p"]

        x = data["x"][0]
        y = data["y"][0]

        x_ps = phasespace["x"]
        y_ps = phasespace["y"]

        plt.scatter(x_ps, y_ps, c="grey", s=0.1)
        plt.scatter(x, y, c="blue", s=1)
        plt.xlabel("X", fontsize=18)
        plt.ylabel("Y", fontsize=18)
        plt.title(rf"$\nu_\text{{m}} = {par.nu_m:.2f}, a \omega_\text{{m}} = {par.epsilon*par.omega_s:.2f} \ \text{{s}}^{{-1}}$", fontsize=18)
        plt.axis('square')
        #plt.xlim(-3.5, 3.5)
        #plt.ylim(-3.5, 3.5)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()

        #x_mean = np.mean(x)
        #y_mean = np.mean(y)

        name_dir = f"nu_{par.nu_m:.2f}"
        #name_subdir = f"eps_{par.epsilon:.3f}"

        dir_path = Path(f"./final_data_otherconfig/{config}/{name_dir}")
        dir_path.mkdir(parents=True, exist_ok=True)

        #np.savez(dir_path / "full_distr_isl.npz", x=x, y=y)
        np.savez(f"final_data_otherconfig/ww/{name_dir}/isl_eps_{par.epsilon:.3f}.npz", x=x, y=y)

        plt.show()
    
    if mode == "phasespace":
        data = np.load(f"action_angle/phasespace_{n_particles}_a{par.a:.7f}_nu{par.omega_m/par.omega_s:.5f}_{machine}_{config}.npz")
        data_background = np.load(f"action_angle/phasespace_75_a{par.a:.7f}_nu{par.omega_m/par.omega_s:.5f}_{machine}_{config}.npz")

        x_ps = data_background["x"]
        y_ps = data_background["y"]

        x = data["x"]
        y = data["y"]

        #name_dir = f"eps_{par.epsilon:.3f}"
        name_dir = f"nu_{par.nu_m:.2f}"
        name_subdir = f"eps_{par.epsilon:.3f}"

        #np.savez(f"final_data_otherconfig/ww/eps_0.065/isl_nu_{par.nu_m:.3f}.npz", x=x, y=y)

        plt.scatter(x_ps, y_ps, s=1)
        #plt.scatter(x[-1, :], y[-1, :], s=1)

        #plt.show()
                
# ----------------------------------

if __name__ == "__main__":
    mode = sys.argv[1]
    n_particles = int(sys.argv[2])
    config = sys.argv[3]

    plot(mode, n_particles, config)
