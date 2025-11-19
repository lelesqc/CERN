import sys
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import params
from scipy.integrate import trapezoid
from scipy.stats import wasserstein_distance, ks_2samp

def plot(poincare_mode, n_particles, n_to_plot, idx_start, idx_end, par):
    fn.par = par
    
    a_start = par.a_lambda(par.T_percent)
    omega_start = par.omega_lambda(par.T_percent)
    a_end = par.a_lambda(par.T_tot)
    omega_end = par.omega_lambda(par.T_tot)

    a_start_str = f"{a_start:.3f}"
    omega_start_str = f"{omega_start:.7f}"
    a_end_str = f"{a_end:.3f}"
    omega_end_str = f"{omega_end:.7f}"

    str_title = f"a{a_start_str}-{a_end_str}_nu{float(omega_start_str)/par.omega_s:.7f}-{float(omega_end_str)/par.omega_s:.3f}"

    data = np.load(f"action_angle/{poincare_mode}_{str_title}_{idx_start}_{idx_end}.npz")
    data_ps = np.load(f"../phasespace_stochastic/action_angle/phasespace_100_a0.050_nu0.83_als.npz")
    data_qp = np.load(f"integrator/evolved_qp_{poincare_mode}.npz")

    t_final = data_qp["t_list"]

    x = data['x']
    y = data['y']
    x_ps = data_ps["x"]
    y_ps = data_ps["y"]
    q = data_qp["q"]
    p = data_qp["p"]
    #t_final = data_qp["t_final"]

    mask = ((x+0.5)**2 + y**2) > 9    # ALS 
    #mask = ((x+0.25)**2 + y**2) > 1.5

    """x_rel = x - np.mean(x)
    y_rel = y - np.mean(y)
    X = np.vstack([x_rel, y_rel])            # shape (2, N)
    Sigma = np.cov(X)                # 2x2
    eps_rms = float(np.sqrt(np.linalg.det(Sigma)))
    I = 0.5 * (x_rel**2 + y_rel**2)
    I_mean = float(np.mean(I))
    print(f"Final rms emittance: {eps_rms:.6e}  |  mean action: {I_mean:.6e}")"""

    if poincare_mode in ["first", "last"]:
        x_isl = x[mask]
        y_isl = y[mask]
        x_cen = x[~mask]
        y_cen = y[~mask]
        #q_isl = q[mask]
        #p_isl = p[mask]
        #q_cen = q[~mask]
        #p_cen = p[~mask]

        """plt.scatter(x_ps, y_ps, s=1) 
        plt.scatter(x, y, s=1)
        plt.show()

        par.t = t_final

        E0 = fn.hamiltonian(np.mean(q), np.mean(p))
        #E0 = np.min(fn.hamiltonian(q, p))
        h_0 = fn.hamiltonian(q, p) 
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

        print(E0, np.min(h_0), fn.hamiltonian(0, 0))

        actions, _ = fn.compute_action_angle(kappa_squared, 1)
        actions -= np.min(actions)

        sorted_idx = np.argsort(actions)
        energies_i = h_0[sorted_idx]
        actions_sorted_i = actions[sorted_idx]
    
        # istogramma
        hist, bin_edges = np.histogram(actions, bins=100, density=True)
        P_continuous = np.exp(-(np.interp(actions_sorted_i, actions_sorted_i, energies_i) - E0) / par.temperature)
        Z = trapezoid(P_continuous, actions_sorted_i)
        P_continuous /= Z

        # calcola la media teorica sui bin (integrale / ampiezza bin)
        P_H_bin = np.zeros_like(hist)
        for j in range(len(hist)):
            x0, x1 = bin_edges[j], bin_edges[j+1]
            mask = (actions_sorted_i >= x0) & (actions_sorted_i < x1)
            if np.any(mask):
                P_H_bin[j] = trapezoid(P_continuous[mask], actions_sorted_i[mask]) / (x1 - x0)
            else:
                # se il bin è vuoto, interpola
                P_H_bin[j] = np.interp(0.5*(x0+x1), actions_sorted_i, P_continuous)

        # ora confronta densità media (coerente con istogramma)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        L2 = np.sqrt(trapezoid((hist - P_H_bin)**2, bin_centers))

        plt.hist(actions, bins=100, density=True, alpha=0.5, label="Distr. of actions")
        plt.plot(bin_centers, P_H_bin, label="Boltz. distribution")
        plt.title(f"L2 norm: {L2:.2f}")
        plt.legend()
        plt.xlabel("I")
        plt.ylabel(r"$\rho(I)$")
        #plt.savefig("../results/resonance11/center/last_hist_adiab_fcc.png")
        plt.show()   

        par.t = 0
             
        init_data = np.load("../phasespace_stochastic/integrator/evolution_qp_10000_fcc.npz")
        q_init = init_data["q"]
        p_init = init_data["p"]

        E0_in = fn.H0_for_action_angle(np.mean(q_init), np.mean(p_init))
        h_0_in = fn.H0_for_action_angle(q_init, p_init) 
        kappa_squared_in = 0.5 * (1 + h_0_in / (par.A**2))

        actions_init, _ = fn.compute_action_angle(kappa_squared_in, 1)

        sorted_idx = np.argsort(actions_init)
        energies_i_in = h_0_in[sorted_idx]
        actions_sorted_i_in = actions_init[sorted_idx]
    
        # istogramma
        hist_in, bin_edges_in = np.histogram(actions_init, bins=100, density=True)
        P_continuous_in = np.exp(-(np.interp(actions_sorted_i_in, actions_sorted_i_in, energies_i_in) - E0_in) / par.temperature)
        Z_in = trapezoid(P_continuous_in, actions_sorted_i_in)
        P_continuous_in /= Z_in

        # calcola la media teorica sui bin (integrale / ampiezza bin)
        P_H_bin_in = np.zeros_like(hist_in)
        for j in range(len(hist_in)):
            x0, x1 = bin_edges_in[j], bin_edges_in[j+1]
            mask = (actions_sorted_i_in >= x0) & (actions_sorted_i_in < x1)
            if np.any(mask):
                P_H_bin_in[j] = trapezoid(P_continuous_in[mask], actions_sorted_i_in[mask]) / (x1 - x0)
            else:
                # se il bin è vuoto, interpola
                P_H_bin_in[j] = np.interp(0.5*(x0+x1), actions_sorted_i_in, P_continuous_in)

        # ora confronta densità media (coerente con istogramma)
        bin_centers_in = 0.5 * (bin_edges_in[:-1] + bin_edges_in[1:])
        L2_in = np.sqrt(trapezoid((hist_in - P_H_bin_in)**2, bin_centers_in))


        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        # Primo istogramma (iniziale)
        axes[0].hist(actions_init, bins=100, density=True, alpha=0.5, label="Distr. of actions")
        axes[0].plot(bin_centers_in, P_H_bin_in, label="Boltz. distribution")
        axes[0].set_title(f"First hist\nL2 norm: {L2_in:.2f}")
        axes[0].set_xlabel("I")
        axes[0].set_ylabel(r"$\rho(I)$")
        axes[0].legend()

        # Secondo istogramma (finale)
        axes[1].hist(actions, bins=100, density=True, alpha=0.5, label="Distr. of actions")
        axes[1].plot(bin_centers, P_H_bin, label="Boltz. distribution")
        axes[1].set_title(f"Final hist\nL2 norm: {L2:.2f}")
        axes[1].set_xlabel("Actions")
        axes[1].legend()

        #plt.savefig("../results/resonance11/center/proof_adiab_inv_distr_actions_fcc.png")

        plt.show()"""

        r"""par.t = t_final

        h_0_isl = fn.H0_for_action_angle(q_isl, p_isl)
        k2_isl = 0.5 * (1 + h_0_isl / (par.A**2))

        h0_min_isl = fn.H0_for_action_angle(np.mean(q_isl), np.mean(p_isl))
        k2_min_isl = 0.5 * (1 + h0_min_isl / (par.A**2))

        E0_isl_altor = fn.hamiltonian(np.mean(q_isl), np.mean(p_isl))
        E0_isl = fn.H_resonant(np.mean(q_isl), np.mean(p_isl), np.mean(x_isl), np.mean(y_isl), np.mean(x_isl), np.mean(y_isl), k2_min_isl)

        energies_isl = []
        #energies_isl = fn.hamiltonian(q_isl, p_isl)
        for i in range(q_isl.shape[0]):
            energies_isl.append(fn.H_resonant(q_isl[i], p_isl[i], x_isl[i], y_isl[i], np.mean(x_isl), np.mean(y_isl), k2_isl[i]))
            
        avg_energies_isl = np.mean(energies_isl - E0_isl)

        print(E0_isl, np.min(energies_isl))


        E0_cen = fn.hamiltonian(np.mean(q_cen), np.mean(p_cen))
        energies_cen = fn.hamiltonian(q_cen, p_cen)
        avg_energies_cen = np.mean(energies_cen - E0_cen)

        variances_cen = np.var(q_cen - np.mean(q_cen))
        variances_isl = np.var(q_isl - np.mean(q_isl))

        print(avg_energies_isl, variances_cen, variances_isl, par.temperature)

        plt.hist(energies_cen - E0_cen, bins=50, density=True)
        plt.show()
        plt.hist(energies_isl - E0_isl, bins=50, density=True)
        plt.show()

        I = ((x_isl - np.mean(x_isl))**2 + (y_isl - np.mean(y_isl))**2) / 2
        print(np.std(fn.H0_for_action_angle(q,p)), np.std(I * par.omega_lambda(par.t)))"""   
        print(f"Trapped: {x_isl.shape[0]}, center: {x_cen.shape[0]}")

        n_isl = int(np.count_nonzero(mask))
        n_cen = int(np.count_nonzero(~mask))

        #plt.scatter(x_cen, y_cen, s=5, alpha=1.0, color='C0', label=f"Center (N={n_cen})")
        #plt.scatter(x_isl, y_isl, s=5, alpha=1.0, color='C1', label=f"Island (N={n_isl})")
        #plt.scatter(x_ps, y_ps, s=1)
        #plt.scatter(x, y, s=3)
        #plt.xlabel("X", fontsize=16)
        #plt.ylabel("Y", fontsize=16)
        #plt.show()

        np.savez(f"../results/resonance11/final_results/als/data/damp/{par.nu_m_i:.7f}_{par.nu_m_f:.3f}_{idx_start}_{idx_end}.npz", n_isl=n_isl)
        #plt.xlim(-15, 15)
        #plt.ylim(-15, 15)
        #plt.tick_params(labelsize=18)
        #plt.axis("square")
        #plt.title(rf"Adiabatic trapping for $\nu_m = {par.nu_m_i:.6f} - {par.nu_m_f:.6f}$")
        #plt.savefig(f"../results/resonance11/trapping_hamiltonian/pics/nu_i_{par.nu_m_i:.7f}_nu_f_{par.nu_m_f:.3f}_als.png")
        #plt.close()
        #np.savez(f"../results/resonance11/trapping_stochastic/data/nu_i_{par.nu_m_i:.7f}_nu_f_{par.nu_m_f:.3f}_{idx_start}_{idx_end}_als.npz",
        # n_isl=n_isl, n_cen=n_cen)
        #np.savez(f"../results/resonance11/trapping_stochastic/data/nu_i_{par.nu_m_i:.7f}_nu_f_{par.nu_m_f:.3f}_als.npz", n_isl=n_isl, n_cen=n_cen)
        #plt.tight_layout()
        #plt.show()

    elif poincare_mode == "all":
        """n_sections = len(x) // n_particles
        idx_to_plot = np.linspace(0, n_sections-1, n_to_plot, dtype=int)

        _, axes = plt.subplots(1, n_to_plot, figsize=(4*n_to_plot, 4), sharex=True, sharey=True)
        if n_to_plot == 1:
            axes = [axes]
        for ax, i in zip(axes, idx_to_plot):
            ax.scatter(x[i*n_particles:(i+1)*n_particles], y[i*n_particles:(i+1)*n_particles], s=2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")"""
        
        """actions = np.zeros((q.shape[0], q.shape[1]))
        L2_list = []

        temperature = par.temperature

        n_times = q.shape[0]

        actions = np.empty((n_times, n_particles))
        energies = []
        kappas = []

        step = 1

        for i, idx in enumerate(range(0, n_times, step)):
            h_0 = fn.H0_for_action_angle(q[idx, :], p[idx, :])

            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
            kappas.append(kappa_squared)

            actions[i, :], _ = fn.compute_action_angle(kappa_squared, 1)
            
            h0_of_I = 2 * par.A**2 * (kappa_squared - 1/2)
            energies.append(h0_of_I)

        E0 = fn.hamiltonian(np.mean(q[-1, :]), np.mean(p[-1, :]))
        print(energies[-1] - E0)
        plt.hist(energies[-1] - E0, bins=100)
        plt.show()

        for i in range(0, n_times, step):
            nan_indices = np.where(np.isnan(actions[i, :]))[0]
            if nan_indices.size > 0:
                print(f"NaN in actions[{i},:] agli indici: {nan_indices}")
                print("Valori corrispondenti di q:", q[i, :][nan_indices])
                print("Valori corrispondenti di p:", p[i, :][nan_indices])
                print(kappas[i][nan_indices])
            energies_i = energies[i]
            sorted_idx = np.argsort(actions[i, :])
            energies_i = energies_i[sorted_idx]
            actions_sorted_i = actions[i, :][sorted_idx]

            E0 = fn.hamiltonian(np.mean(q[idx, :]), np.mean(p[idx, :]))
        
            # istogramma
            hist, bin_edges = np.histogram(actions[i, :], bins=100, density=True)
            P_continuous = np.exp(-(np.interp(actions_sorted_i, actions_sorted_i, energies_i) - E0) / temperature)
            Z = trapezoid(P_continuous, actions_sorted_i)
            P_continuous /= Z

            # calcola la media teorica sui bin (integrale / ampiezza bin)
            P_H_bin = np.zeros_like(hist)
            for j in range(len(hist)):
                x0, x1 = bin_edges[j], bin_edges[j+1]
                mask = (actions_sorted_i >= x0) & (actions_sorted_i < x1)
                if np.any(mask):
                    P_H_bin[j] = trapezoid(P_continuous[mask], actions_sorted_i[mask]) / (x1 - x0)
                else:
                    # se il bin è vuoto, interpola
                    P_H_bin[j] = np.interp(0.5*(x0+x1), actions_sorted_i, P_continuous)

            # ora confronta densità media (coerente con istogramma)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            L2 = np.sqrt(trapezoid((hist - P_H_bin)**2, bin_centers))
            L2_list.append(L2)

        fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

        # Primo istogramma (iniziale)
        axes[0].hist(actions[0, :], bins=100, density=True, alpha=0.5, label="Distr. of actions")
        axes[0].plot(bin_centers, P_H_bin, label="Boltz. distribution")
        axes[0].set_title(f"First hist\nL2 norm: {L2_list[0]:.2f}")
        axes[0].set_xlabel("Actions")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        # Secondo istogramma (finale)
        axes[1].hist(actions[actions.shape[0] // 2, :], bins=100, density=True, alpha=0.5, label="Distr. of actions")
        axes[1].plot(bin_centers, P_H_bin, label="Boltz. distribution")
        axes[1].set_title(f"Mid hist\nL2 norm: {L2_list[actions.shape[0]//2]:.2f}")
        axes[1].set_xlabel("Actions")
        axes[1].legend()

        axes[2].hist(actions[actions.shape[0] // 2, :], bins=100, density=True, alpha=0.5, label="Distr. of actions")
        axes[2].plot(bin_centers, P_H_bin, label="Boltz. distribution")
        axes[2].set_title(f"Last hist\nL2 norm: {L2_list[-1]:.2f}")
        axes[2].set_xlabel("Actions")
        axes[2].legend()

        plt.tight_layout()
        plt.show()"""
        

        """for i in range(0, actions.shape[0], 100):
            plt.hist(actions[i, :], bins=100, density=True)

            plt.show()"""
        
        thetas = data["theta"]
        actions = data["actions"]
        thetaz = thetas[:, 0]
        actionz = actions[:, 0]

        data_qp = np.load(f"integrator/evolved_qp_{poincare_mode}_{idx_start}_{idx_end}.npz")

        t_final = data_qp["t_list"]
        print(t_final)

        timez = np.linspace(0, t_final, thetaz.shape[0])
        psis = []
        
        listez = np.linspace(0, t_final, thetaz.shape[0])
        plt.scatter(timez, thetaz, s=5)
        plt.ylim(0, 4)
        plt.show()

        thetaz = thetas[:, 1]

        plt.scatter(timez, thetaz, s=1)
        plt.show()

        plt.scatter(x_ps, y_ps, s=1)
        plt.scatter(x[-1, :], y[-1, :], s=1)
        plt.show()

    


def plot_skrt(poincare_mode, n_particles, n_to_plot):
    a_start = par.a_lambda(par.T_percent)
    omega_start = par.omega_lambda(par.T_percent)
    a_end = par.a_lambda(par.T_tot)
    omega_end = par.omega_lambda(par.T_tot)

    a_start_str = f"{a_start:.3f}"
    omega_start_str = f"{omega_start:.2f}"
    a_end_str = f"{a_end:.3f}"
    omega_end_str = f"{omega_end:.2f}"

    str_title = f"a{a_start_str}-{a_end_str}_nu{float(omega_start_str)/par.omega_s:.2f}-{float(omega_end_str)/par.omega_s:.2f}"

    data = np.load(f"action_angle/{poincare_mode}_{str_title}.npz")

    x = data['x']
    y = data['y']

    if poincare_mode in ["first", "last"]:
        plt.scatter(x, y)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.show()
    elif poincare_mode == "all":
        time_array = np.linspace(0, par.T_tot, par.n_steps)
        print(time_array)
        
        a_values = np.array([par.a_lambda(t) for t in time_array])
        omega_values = np.array([par.omega_lambda(t) for t in time_array])
        
        plt.figure(figsize=(12, 8))
        
        ax1 = plt.subplot(2, 1, 1)
        color1 = 'tab:red'
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('a', color=color1)
        line1 = ax1.plot(time_array, a_values, color=color1, label='a(t)')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel(r'$\nu_m$', color=color2)
        line2 = ax2.plot(time_array, omega_values/par.omega_s, color=color2, label=r'$\nu_m(t)$')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc=(0.02, 0.7))
        
        plt.title(r'Temporal evolution of $a$ e $\nu_m$')
        
        plt.subplot(2, 1, 2)
        n_sections = len(x) // n_particles
        idx_to_plot = np.linspace(0, n_sections-1, min(n_to_plot, n_sections), dtype=int)
        
        for i in idx_to_plot:
            plt.scatter(x[i*n_particles:(i+1)*n_particles], 
                       y[i*n_particles:(i+1)*n_particles], 
                       s=2, alpha=0.7, label=f'Sezione {i+1}')
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Sezioni di Poincaré")
        if len(idx_to_plot) <= 5:  # Mostra legenda solo se poche sezioni
            plt.legend()
        
        plt.tight_layout()
        plt.show()


# -----------------------------------------


if __name__ == "__main__":
    poincare_mode = str(sys.argv[1])
    n_particles = int(sys.argv[2])
    n_to_plot = int(sys.argv[3])
    idx_start = int(sys.argv[4])
    idx_end = int(sys.argv[5])
    params_path = sys.argv[6] if len(sys.argv) > 6 else "params.yaml"
    par = params.load_params(params_path)

    plot(poincare_mode, n_particles, n_to_plot, idx_start, idx_end, par)
