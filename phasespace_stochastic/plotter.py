import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
#import alphashape

import functions as fn
import params_fcc

par = params_fcc.Params()

def plot(mode):
    if mode == "evolution":
        data = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}_{par.sigma:.3f}.npz")
        integrator = np.load("integrator/evolved_qp_evolution.npz")

        x = data["x"]
        y = data["y"]
      
        """plt.hist(delta_E, bins=25, density=True, alpha=0.5, label="Simulazione")
        plt.plot(bin_centers, exp_func(bin_centers, T_fit, norm_fit), 'r-', lw=2, label=fr"Fit $T={T_fit:.2f}$")
        plt.title(f"Temperature: {T_fit:.1f}")
        plt.xlabel("Energies", fontsize=20)
        plt.ylabel("Probability Density", fontsize=20)
        #plt.yscale("log")
        #plt.plot(energies_neg, P_H, 'r-', lw=2, label="Teorica $e^{-(E-E_0)/T_{eff}}$")
        plt.show()
        """
        ps = np.load("./action_angle/phasespace_a0.050_nu0.80_extra_fcc.npz")
        x_ps = ps["x"]
        y_ps = ps["y"]
        
        #mask = (x+0.2)**2 + y**2 > 1.5**2
        #x_mask = x[mask]
        #y_mask = y[mask]

        n_gt = int(np.count_nonzero(x > 0.5))
        n_lt = int(np.count_nonzero(x < 0.5))
        n_eq = int(np.count_nonzero(x == 0.5))
        total = x.size

        print(f"Particles: total={total}, x>0.5={n_gt}, x<0.5={n_lt}, x==0.5={n_eq}")

        # opzionale: mostra i numeri sul grafico
        info = f"total={total}\nx>0.5={n_gt}\nx<0.5={n_lt}"

        np.savez(f"./infos/trapping_results_{par.sigma:.3f}.npz", trap=n_gt, center=n_lt)
        plt.annotate(info, xy=(0.98, 0.02), xycoords="axes fraction",
                     ha="right", va="bottom", fontsize=9,
                     bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        plt.scatter(x_ps, y_ps, s=1, label="Phase space")
        plt.scatter(x, y, s=1, label="Distribution")
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("square")
        plt.savefig(f"../results/phasespaces/ps_a0.050_nu0.80_sigma_{par.sigma}.png")
        plt.close()

    if mode == "phasespace":
        action_angle = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}_change.npz")

        x = action_angle["x"]
        y = action_angle["y"]

        """i_max, j_max = np.unravel_index(np.argmax(x, axis=None), x.shape)
        # j_max è l'indice della particella (seconda dimensione)
        print("step:", i_max, "particle:", j_max, "x_max:", x[i_max, j_max])

        xy_j = np.vstack((x[:, 949], y[:, 949])).T    
        hull_j = alphashape.alphashape(xy_j, alpha=0.1)
        area_island = hull_j.area
        print("area:", area_island)
        
        if hull_j is not None:
            if hasattr(hull_j, "geoms"):  # MultiPolygon
                for geom in hull_j.geoms:
                    x_hull, y_hull = geom.exterior.xy
                    plt.plot(x_hull, y_hull, c="r", label="Hull")
            else:  # Polygon
                x_hull, y_hull = hull_j.exterior.xy
                plt.plot(x_hull, y_hull, c="r", label="Hull")

        maskz = (y > -0.1) & (y < 0.1)
        masked_x = x[maskz]

        # valore minimo e indici (step, particle) corrispondenti
        min_x = masked_x.min()
        coords = np.argwhere(maskz)                # array di (step, particle) per ogni True
        min_idx_in_mask = int(np.argmin(masked_x)) # indice nell'array masked_x / coords
        min_coords = coords[min_idx_in_mask:min_idx_in_mask+1]

        print(f"Min x sotto maskz: {min_x}")
        print(f"Indici (step, particle): {min_coords}")"""      

        plt.scatter(x, y, s=1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Phase space")
        plt.axis("square")
        #plt.savefig(f"../results/phasespaces/ps_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}_fcc.png", dpi=200)
        plt.show()


# ----------------------------------


if __name__ == "__main__":
    mode = sys.argv[1]

    plot(mode)
