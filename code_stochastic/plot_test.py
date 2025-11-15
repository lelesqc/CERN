import numpy as np
import matplotlib.pyplot as plt
import params as par
import os
import functions as fn
from tqdm.auto import tqdm

def plot_test():
    data = np.load("stochastic_studies/adiab_invariant/vars_and_avg_energies.npz")

    vars = data["vars"]
    energies = data["energies"]
    mean = []

    times = np.linspace(0, par.T_tot, len(energies))

    for i in range(len(energies)):
        mean.append(np.mean(energies[i, :]))

    plt.scatter(times, mean, s=1) 
    plt.show()

def altro():
    folder = "./stochastic_studies/adiab_invariant/ang_coeff_vs_exc_amplitude"
    epsilon_f_list = []
    slope_list = []

    for fname in os.listdir(folder):
        if fname.endswith("fcc.npz"):
            data = np.load(os.path.join(folder, fname))
            epsilon_f_list.append(data["epsilon_f"].item() / par.nu_m_f)
            slope_list.append(data["slope"].item())

    plt.scatter(epsilon_f_list[:-1], slope_list[:-1], s=20)
    plt.xlabel("Modulation amplitude")
    plt.ylabel("Slope")
    plt.yscale("log")
    #plt.savefig("../results/resonance11/center/slope_vs_a_mean_energy_fcc.png")
    plt.show()

#%%

import numpy as np
import os
import matplotlib.pyplot as plt

root_dir = "../results/resonance11/trapping_hamiltonian/data"

n_isl_ham = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.endswith(".npz"):
            fpath = os.path.join(dirpath, fname)
            try:
                d = np.load(fpath)
                if "n_isl" in d.files and "n_cen" in d.files:
                    n_isl_ham.append(d["n_isl"].item()/100 if d["n_isl"].ndim == 0 else d["n_isl"])
            except Exception:
                pass

root_dir = "../results/resonance11/trapping_stochastic/data"

n_isl_stoc = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.endswith(".npz"):
            fpath = os.path.join(dirpath, fname)
            try:
                d = np.load(fpath)
                if "n_isl" in d.files and "n_cen" in d.files:
                    n_isl_stoc.append(d["n_isl"].item()/100 if d["n_isl"].ndim == 0 else d["n_isl"])
            except Exception:
                print("ksrt")
                pass


k = 7
val = 100.0
n_isl_ham.extend([val] * k)

print(len(n_isl_stoc))
print(len(n_isl_ham))

list_idx = np.linspace(0.9597, 0.9637, len(n_isl_ham))
plt.scatter(list_idx[1:], n_isl_ham[1:], s=10, label="Hamiltonian")
plt.scatter(list_idx[1:], n_isl_stoc[1:], s=10, label="Stochastic")
plt.title(r"$\nu_\text{m, f}$ = 0.83")
plt.xlabel(r"$\nu_\text{m, i}$")
plt.ylabel("Trapping probability")
plt.legend()
#plt.savefig("../results/resonance11/trapping_results/pics/stoc_vs_ham_FCC_Z.png")
plt.show()
#np.savez("../results/resonance11/trapping_results/data/stoc_vs_ham_FCC_Z.npz", list_nu_m=list_idx, trap_prob_ham=n_isl_ham, trap_prob_stoc=n_isl_stoc)

#%%

if __name__ == "__main__":
    plot_test()
    #altro_ancora()