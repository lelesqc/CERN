import sys
import numpy as np
import matplotlib.pyplot as plt

import params as par

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

    x = data['x']
    y = data['y']

    if poincare_mode in ["first", "last"]:
        plt.figure(figsize=(7,7))
        plt.scatter(x, y, s=3, label=r"Final distribution", alpha=1.0)
        #plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=16)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.legend(fontsize=20)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()
    """elif poincare_mode == "all":
        n_sections = len(x) // n_particles
        idx_to_plot = np.linspace(0, n_sections-1, n_to_plot, dtype=int)

        _, axes = plt.subplots(1, n_to_plot, figsize=(4*n_to_plot, 4), sharex=True, sharey=True)
        if n_to_plot == 1:
            axes = [axes]
        for ax, i in zip(axes, idx_to_plot):
            ax.scatter(x[i*n_particles:(i+1)*n_particles], y[i*n_particles:(i+1)*n_particles], s=2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y") """

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
    
    plot_skrt(poincare_mode, n_particles, n_to_plot)
