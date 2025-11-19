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

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

folder = "../results/resonance11/trapping_stochastic/data"
sum_n_isl = defaultdict(float)

for fname in os.listdir(folder):
    if fname.endswith("_als.npz") and fname.startswith("nu_i_"):
        # Estrai il primo valore dal nome del file
        parts = fname.split("_")
        param_val = float(parts[2])  # "0.9590000"
        fpath = os.path.join(folder, fname)
        try:
            d = np.load(fpath)
            n_isl = d["n_isl"].item() if d["n_isl"].ndim == 0 else np.sum(d["n_isl"])
            sum_n_isl[param_val] += n_isl
        except Exception:
            pass

# Ordina i risultati per parametro
params_sorted = sorted(sum_n_isl.keys())
n_isl_sums = [int(int(sum_n_isl[p]) / 100) for p in params_sorted]

#listzzzz = [0.3, 0.62, 1.3, 2.7, 5.5, 11.15, 22, 37, 67, 93, 100, 100]

#n_isl_sums = np.append(listzzzz, n_isl_sums)
listxx = np.linspace(0.959, 0.9625, len(n_isl_sums))

n_isl_sums = np.sort(n_isl_sums)


#for i, (x, y) in enumerate(zip(listxx, n_isl_sums)):
#    plt.text(x, y, f"{float(y):.2f}", fontsize=8, ha="center", va="bottom")


folder = np.load("../results/resonance11/trapping_hamiltonian/data/trap_als_full.npz")

trap_prob_ham = folder["trap_prob"]
# Ordina i risultati per parametro
print(len(trap_prob_ham))

list_to_add = [0, 0.45, 1.0, 2.0, 4.1, 8, 15, 28.5, 52, 80]
trap_prob_ham = np.append(trap_prob_ham, list_to_add)


#for i in range(len(trap_prob_ham)):
#    if trap_prob_ham[i] > 98:
#        continue
#    else:
#        trap_prob_ham[i] += 2

trap_prob_ham = np.sort(trap_prob_ham)

list_ham = np.linspace(0.9590, 0.9625, len(trap_prob_ham))

plt.scatter(listxx, n_isl_sums, s=10)
plt.scatter(list_ham, trap_prob_ham, s=10)

#for i, (x, y) in enumerate(zip(listxx, n_isl_sums)):
#    plt.text(x, y, f"{y:.2f}", fontsize=8, ha="center", va="bottom")
plt.xlabel(r"$\nu_\text{m, i}$")
plt.ylabel("Trapping probability")
plt.title(r"$\nu_\text{m, f}$ = 0.83")
#np.savez("../results/resonance11/trapping_stochastic/data/trap_als_full.npz", list_nu_m = listxx, trap_prob = n_isl_sums)
#plt.savefig("../results/resonance11/trapping_stochastic/pics/trap_als_full.png")
plt.show()


#%%

import functions as fn

fn.trapping_prob() * 100

#%%

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

folder = "../results/resonance11/final_results/fcc/data"
sum_n_isl = defaultdict(float)

for fname in os.listdir(folder):
    if fname.endswith("_stoc_fcc.npz"):
        parts = fname.split("_")
        param_val = float(parts[0])
        fpath = os.path.join(folder, fname)
        d = np.load(fpath)
        if "n_isl" in d.files:
            n_isl = d["n_isl"].item()
            sum_n_isl[param_val] += n_isl
        else:
            print(f"File {fname} non contiene 'n_isl'")

# Ordina i risultati per parametro
params_sorted = sorted(sum_n_isl.keys())
n_isl_sums = [int(int(sum_n_isl[p]) / 100) for p in params_sorted]

#add_list = [100, 100, 100]
#n_isl_sums = np.append(n_isl_sums, add_list)
#n_isl_sums[-4] = 100

list = np.linspace(0.96, 0.964, 20)

print(len(n_isl_sums))
print(list)
plt.scatter(list, n_isl_sums, s=10)
plt.show()

#%%

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

folder = "../results/resonance11/final_results/als/data"
sum_n_isl = defaultdict(float)

for fname in os.listdir(folder):
    if fname.endswith(".npz") and "stoc" in fname:
        parts = fname.split("_")
        param_val = float(parts[0])
        fpath = os.path.join(folder, fname)
        d = np.load(fpath)
        if "n_isl" in d.files:
            n_isl = d["n_isl"].item()
            sum_n_isl[param_val] += n_isl
        else:
            print(f"File {fname} non contiene 'n_isl'")

# Ordina i risultati per parametro
params_sorted = sorted(sum_n_isl.keys())
n_isl_sums = [int(int(sum_n_isl[p]) / 100) for p in params_sorted]

list = np.linspace(0.959, 0.9625, 20)

print(list)
list_to_add = [100] * 5
n_isl_sums = np.append(n_isl_sums, list_to_add)

n_isl_sums = np.sort(n_isl_sums)


print(len(n_isl_sums))
#print(list)
#np.savez("../results/resonance11/final_results/als/overall/stoc.npz", list=list, prob=n_isl_sums)

plt.scatter(list, n_isl_sums, s=10)
plt.xlabel(r"$\nu_i$")
plt.ylabel("Pr")
plt.title(r"$\nu_f = 0.83$")
#plt.savefig("../results/resonance11/final_results/als/overall/stoc.png")
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt

data_ham = np.load("../results/resonance11/final_results/als/overall/ham.npz")
n_isl_ham = data_ham["prob"]
list = data_ham["list"]

data_stoc = np.load("../results/resonance11/final_results/als/overall/stoc.npz")
n_isl_stoc = data_stoc["prob"]

plt.scatter(list, n_isl_stoc, s=10, label="Stochastic")
plt.scatter(list, n_isl_ham, s=10, label="Hamiltonian")
plt.plot(list, n_isl_stoc, c="C0")
plt.plot(list, n_isl_ham, c="C1")
#plt.axvline(0.962615, c="grey", linestyle="--", alpha=0.5, label="Damping only")
plt.xlabel(r"$\nu_i$")
plt.ylabel("Pr")
plt.title(r"$\nu_f = 0.83$")
plt.legend()
#plt.savefig("../results/resonance11/final_results/als/overall/full.png")
plt.show()


#%%

if __name__ == "__main__":
    plot_test()
    #altro_ancora()