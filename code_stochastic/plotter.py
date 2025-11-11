import sys
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import params_fcc as par
from scipy.integrate import trapezoid
from scipy.stats import wasserstein_distance, ks_2samp

def plot(poincare_mode, n_particles, n_to_plot):
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
    data_ps = np.load(f"../phasespace_stochastic/action_angle/phasespace_100_a0.050_nu0.80_fcc.npz")
    data_qp = np.load(f"integrator/evolved_qp_{poincare_mode}.npz")

    x = data['x']
    y = data['y']
    x_ps = data_ps["x"]
    y_ps = data_ps["y"]
    q = data_qp["q"]
    p = data_qp["p"]
    #t_final = data_qp["t_final"]

    mask = ((x+0.25)**2 + y**2) > 4.2
    #mask = ((x+0.8)**2 + y**2) > 56

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
        q_isl = q[mask]
        p_isl = p[mask]
        q_cen = q[~mask]
        p_cen = p[~mask]

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
        print(f"Trapped: {q_isl.shape[0]}, center: {q_cen.shape[0]}")

        n_isl = int(np.count_nonzero(mask))
        n_cen = int(np.count_nonzero(~mask))

        print(q, p)


        plt.scatter(x_ps, y_ps, s=1, color='lightgray', label='_nolegend_')
        plt.scatter(x_cen, y_cen, s=3, alpha=1.0, color='C0', label=f"Center (N={n_cen})")
        plt.scatter(x_isl, y_isl, s=3, alpha=1.0, color='C1', label=f"Island (N={n_isl})")

        plt.xlabel("X", fontsize=16)
        plt.ylabel("Y", fontsize=16)
        #plt.xlim(-15, 15)
        #plt.ylim(-15, 15)
        plt.legend(fontsize=14, loc="best")
        plt.tick_params(labelsize=18)
        plt.axis("square")
        plt.title(rf"Adiabatic trapping for $\nu_m = {par.nu_m_i:.2f} - {par.nu_m_f:.2f}$")
        #plt.savefig(f"../results/resonance11/trapping_hamiltonian/nu_i_{par.nu_m_i:.2f}_nu_f_{par.nu_m_f:.2f}_fcc.png")
        #plt.close()
        #np.savez(f"../results/resonance11/trapping_hamiltonian/data/nu_i_{par.nu_m_i:.2f}_nu_f_{par.nu_m_f:.2f}_fcc.npz", n_isl=n_isl, n_cen=n_cen)
        #plt.tight_layout()
        plt.show()

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
        
        actions = np.zeros((x.shape[0], x.shape[1]))
        L2_list = []

        temperature = par.temperature

        dt_plot = 20  # intervallo temporale (in passi di indice)
        M = 500       # numero di bootstrap
        I_t0 = None

        t_list, W_list, ci_low_list, ci_high_list, KS_p_list = [], [], [], [], []

        from tqdm.auto import tqdm

        for i in tqdm(range(0, actions.shape[0], dt_plot)):
            # calcolo punto fisso istantaneo (centro del moto)
            fixed_pt_x = np.mean(x[i, :])
            fixed_pt_y = np.mean(y[i, :])

            # calcolo azioni istantanee (shift rispetto al centro)
            actions[i, :] = ((x[i, :] - fixed_pt_x)**2 + (y[i, :] - fixed_pt_y)**2) / 2

            # azione di riferimento e al tempo attuale
            if I_t0 is None:
                I_t0 = actions[i, :].copy()   # primo step
                continue

            I_tk = actions[i, :]

            # Wasserstein osservato e KS test
            W_obs = wasserstein_distance(I_t0, I_tk)
            _, KS_p = ks_2samp(I_t0, I_tk)

            # bootstrap per intervallo di confidenza su W
            N = len(I_t0)
            Ws = []
            for _ in range(M):
                a = np.random.choice(I_t0, N, replace=True)
                b = np.random.choice(I_tk, N, replace=True)
                Ws.append(wasserstein_distance(a, b))
            ci_low, ci_high = np.percentile(Ws, [2.5, 97.5])

            # accumulo risultati
            t_current = i / actions.shape[0] * par.T_tot
            t_list.append(t_current)
            W_list.append(W_obs)
            ci_low_list.append(ci_low)
            ci_high_list.append(ci_high)
            KS_p_list.append(KS_p)

        # conversione in array per plotting
        t_array = np.array(t_list)
        W_array = np.array(W_list)
        ci_low_array = np.array(ci_low_list)
        ci_high_array = np.array(ci_high_list)
        KS_p_array = np.array(KS_p_list)

        # === Plot finale ===
        plt.figure(figsize=(7, 4))
        plt.plot(t_array, W_array, 'o-', lw=1.5, label='W(t)')
        plt.fill_between(t_array, ci_low_array, ci_high_array, color='C0', alpha=0.25, label='CI bootstrap (95%)')
        plt.xlabel('t')
        plt.ylabel('Wasserstein-1 distance')
        plt.title('Invarianza della distribuzione di azione')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # (facoltativo) Plot p-value KS per completezza
        plt.figure(figsize=(7, 3))
        plt.plot(t_array, KS_p_array, 's-', color='C2')
        plt.axhline(0.05, ls='--', color='k', alpha=0.5)
        plt.xlabel('t')
        plt.ylabel('KS p-value')
        plt.title('Test KS fra distribuzioni successive')
        plt.tight_layout()
        plt.show()


        """for i in range(0, actions.shape[0], 100):
            plt.hist(actions[i, :], bins=100, density=True)

            plt.show()"""

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
    
    plot(poincare_mode, n_particles, n_to_plot)
