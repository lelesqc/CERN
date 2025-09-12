import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import functions as fn
import params as par

def plot(mode):
    """if mode == "phasespace":
        data = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

        x = data['x']
        y = data['y']

        plt.figure(figsize=(7,7))
        plt.scatter(x, y, s=3, label=r"Phase Space for final distr.", alpha=1.0)
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=20)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.legend(fontsize=18)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()"""

    if mode == "phasespace":
        #data = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
        integrator = np.load("integrator/evolved_qp_phasespace.npz")

        q = integrator['q']
        p = integrator['p']

        #print(q[:10])
        #tune_analysis = np.load(f"tune_analysis/fft_results.npz")

        #tunes_list = tune_analysis['tunes_list']

        #x = data['x']
        #y = data['y']

        #print(tunes_list.size)
        energies = np.array([fn.hamiltonian(q[i], p[i]) for i in range(len(q))])
        energies = np.array(energies)
        energies_neg = energies[energies < 0]
        energies_pos = energies[energies > 0]

        q_center = q[energies < 0]
        p_center = p[energies < 0]

        q0 = np.mean(q_center)
        p0 = np.mean(p_center)

        h0_center = fn.hamiltonian(q0, p0)

        delta_E = energies_neg - h0_center

        #T_eff = np.mean(energies_neg)
        #P_H = np.exp(-(energies_neg - np.min(energies_neg)) / T_eff)

        #P_H /= scipy.integrate.trapezoid(P_H, energies_neg - np.mean(energies_neg))
    
        counts, bin_edges = np.histogram(delta_E, bins=25, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        def exp_func(E, T, norm):
            return norm * np.exp(-E / T)
        
        popt, pcov = curve_fit(exp_func, bin_centers, counts, p0=[np.std(delta_E), np.max(counts)])

        T_fit, norm_fit = popt

        print(T_fit)

        plt.hist(delta_E, bins=25, density=True, alpha=0.5, label="Simulazione")
        plt.plot(bin_centers, exp_func(bin_centers, T_fit, norm_fit), 'r-', lw=2, label=fr"Fit $T={T_fit:.2f}$")
        plt.title(f"Temperature: {T_fit:.1f}")
        plt.xlabel("Energies", fontsize=20)
        plt.ylabel("Probability Density", fontsize=20)
        #plt.yscale("log")
        #plt.plot(energies_neg, P_H, 'r-', lw=2, label="Teorica $e^{-(E-E_0)/T_{eff}}$")
        plt.show()

        """plt.figure(figsize=(7,7))
        plt.scatter(x, y, s=3, c='blue', alpha=1.0, label="Phase Space")
        #scatter = plt.scatter(x, y, s=3, c=tunes_list, cmap='viridis', alpha=1.0)
        #plt.colorbar(scatter, label='Tune')
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=20)
        plt.xlim(-15, 15)
        plt.ylim(-5, 5)
        #plt.title("Phase Space colored by Tune", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()"""

    if mode == "tune":
        action_angle = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
        actions = action_angle['actions_list']

        """
        data = np.load("tune_analysis/fft_results.npz")

        x = action_angle['x']

        x_init = x[0, :]            
        actions_init = actions[0, :] 
        
        tunes_list = data['tunes_list']

        actions_init_pos = actions_init
        tunes_list_pos = tunes_list

        print(x.shape)

        for i in range(len(actions_init_pos)):
            print(f"action: {actions_init_pos[i]:.3f}, tune: {tunes_list_pos[i]:.6f}")

        plt.scatter(x_init, tunes_list)"""
        integrator = np.load("integrator/evolved_qp_tune.npz")

        q = integrator['q']
        p = integrator['p']

        step = 50

        energies = [fn.hamiltonian(q[i, 1], p[i, 1]) for i in range(len(range(q.shape[0])))][::step]
        times = np.linspace(0, 100, len(energies))

        indices = np.arange(len(energies))
        plt.scatter(times, energies, c=indices, cmap='viridis')
        plt.colorbar(label='Indice')
        #plt.ylim(-1000, 500)
        plt.show() 

        #plt.scatter(times, actions[::step, 1])
        #plt.ylim(0, 100)
        #plt.show()


# ----------------------------------


if __name__ == "__main__":
    mode = sys.argv[1]

    plot(mode)
