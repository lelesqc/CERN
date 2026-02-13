import os
import importlib
import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

machine = os.environ.get("MACHINE").lower()

def plot(mode, n_particles):
    if mode == "evolution":
        data = np.load(f"action_angle/evolution_{n_particles}_a{par.a:.7f}_nu{par.omega_m/par.omega_s:.5f}_{machine}.npz")
        phasespace = np.load(f"action_angle/phasespace_75_a{par.a:.7f}_nu{par.omega_m/par.omega_s:.5f}_{machine}.npz")

        data_qp = np.load(f"integrator/{mode}_qp_{n_particles}_{machine}.npz")
        phasespace_qp = np.load(f"./integrator/phasespace_qp_150_{machine}.npz")

        q = data_qp["q"]
        p = data_qp["p"]

        x = data["x"][0]
        y = data["y"][0]

        x_ps = phasespace["x"]
        y_ps = phasespace["y"]

        name_dir = f"nu_{par.nu_m:.2f}"
        dir_path = Path(f"./ipac_simulations/a_0.03/{name_dir}")
        dir_path.mkdir(parents=True, exist_ok=True)
        
        #np.savez(f"./ipac_simulations/a_0.03/{name_dir}/relax_point_isl.npz", x=x, y=y)

        plt.scatter(x_ps, y_ps, s=1)
        plt.scatter(x, y, s=3)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

        """x_rel = x - np.mean(x)
        y_rel = y - np.mean(y)
        X = np.vstack([x_rel, y_rel])            # shape (2, N)
        Sigma = np.cov(X)                # 2x2
        eps_rms = float(np.sqrt(np.linalg.det(Sigma)))
        I = 0.5 * (x_rel**2 + y_rel**2)
        I_mean = float(np.mean(I))"""
    
    if mode == "phasespace":
        data = np.load(f"action_angle/phasespace_{n_particles}_a{par.a:.7f}_nu{par.omega_m/par.omega_s:.5f}_{machine}.npz")
        data_qp = np.load(f"integrator/{mode}_qp_{n_particles}_{machine}.npz")
        data_background = np.load(f"action_angle/phasespace_75_a{par.a:.7f}_nu{par.omega_m/par.omega_s:.5f}_{machine}.npz")

        q_ps = data_qp["q"]
        p_ps = data_qp["p"]

        x_ps = data_background["x"]
        y_ps = data_background["y"]

        x = data["x"]
        y = data["y"]

        """name_dir = f"nu_{par.nu_m:.2f}"
        dir_path = Path(f"./ipac_simulations/a_0.03/{name_dir}")
        dir_path.mkdir(parents=True, exist_ok=True)"""

        #np.savez(f"./ipac_simulations/a_0.03/{name_dir}/final_distr_isl.npz", x=x, y=y)

        plt.scatter(x[-1, :], y[-1, :], s=1)
        plt.scatter(x_ps, y_ps, s=1)
        plt.show()
                
# ----------------------------------

if __name__ == "__main__":
    mode = sys.argv[1]
    n_particles = int(sys.argv[2])

    plot(mode, n_particles)
