import numpy as np
import matplotlib.pyplot as plt
import params_fcc as par
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

folder = "../results/resonance11/final_results/fcc/data"
sum_n_isl_stoc = defaultdict(float)
sum_n_isl_ham = defaultdict(float)

for fname in os.listdir(folder):
    if fname.endswith("stoc_article.npz"):
        parts = fname.split("_")
        param_val = float(parts[0])
        fpath = os.path.join(folder, fname)
        d = np.load(fpath)
        if "n_isl" in d.files:
            n_isl = d["n_isl"].item()
            sum_n_isl_stoc[param_val] += n_isl
        else:
            print(f"File {fname} non contiene 'n_isl'")

    elif fname.endswith("ham_article.npz"):
        parts = fname.split("_")
        param_val = float(parts[0])
        fpath = os.path.join(folder, fname)
        d = np.load(fpath)
        if "n_isl" in d.files:
            n_isl = d["n_isl"].item()
            sum_n_isl_ham[param_val] += n_isl
        else:
            print(f"File {fname} non contiene 'n_isl'")


# Ordina i risultati per parametro
params_sorted_stoc = sorted(sum_n_isl_stoc.keys())
params_sorted_ham = sorted(sum_n_isl_ham.keys())

n_isl_stoc = [int(int(sum_n_isl_stoc[p]) / 100) for p in params_sorted_stoc]
n_isl_ham = [int(int(sum_n_isl_ham[p]) / 100) for p in params_sorted_ham]

list = np.linspace(0.9575, 0.9617, 25)

#list_to_add = [100] * 5
#n_isl_sums = np.append(n_isl_sums, list_to_add)

n_isl_stoc = np.sort(n_isl_stoc)
n_isl_ham = np.sort(n_isl_ham)

list_trasl = list.copy()
list_trasl = list - 0.0013

#print(list)
#np.savez("../results/resonance11/final_results/als/overall/stoc.npz", list=list, prob=n_isl_sums)

plt.scatter(list, n_isl_stoc, s=10, label="Stochastic")
plt.scatter(list, n_isl_ham, s=10, label="Hamiltonian")
plt.scatter(list_trasl, n_isl_stoc, s=10)
#plt.plot(list, n_isl_ham, c="C0")
#plt.plot(list_trasl, n_isl_ham, c="C1")
plt.xlabel(r"$\nu_i$")
plt.ylabel("Pr")
plt.title(r"$\nu_f = 0.83$")
plt.xlim(np.min(list), np.max(list))
#plt.savefig("../results/resonance11/final_results/als/overall/stoc.png")
plt.legend()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt

data_ham = np.load("../results/resonance11/final_results/fcc/overall/ham.npz")
n_isl_ham = data_ham["prob"]
list = data_ham["list"]
list_inv = list[::-1]

print(list)
print(list_inv)

data_stoc = np.load("../results/resonance11/final_results/fcc/data/*_article.npz")
n_isl_stoc = data_stoc["prob"]


plt.scatter(list, n_isl_stoc/100, s=22, label="Stochastic")
plt.scatter(list, n_isl_ham/100, s=22, label="Hamiltonian")
#plt.plot(list, n_isl_stoc, c="C0")
#plt.plot(list, n_isl_ham, c="C1")
plt.axvline(0.962625, c="grey", linestyle="--", alpha=0.5, label="Damping only")
plt.xlabel(r"$\nu_i$", fontsize=32)
plt.ylabel("Pr", fontsize=32)
plt.title(r"$\nu_f = 0.83$", fontsize=36)
plt.legend(fontsize=14)
plt.tick_params(labelsize=26)
#plt.savefig("../results/resonance11/final_results/fcc/overall/full_and_damp.png")
plt.show()


#%%

import numpy as np
import functions as fn
from tqdm.auto import tqdm
import params_fcc
import matplotlib.pyplot as plt
par = params_fcc.load_params("params.yaml")  # oppure il file YAML corretto
fn.par = par

data = np.load("./integrator/evolved_qp_all_0_10000.npz")
q = data["q"][:, ::1000]
p = data["p"][:, ::1000]
psi = data["psi"]
t_list = data["t_list"]

#par.dt = par.dt
n_particles = 10
ext = q.shape[0]
int = 5000

print(par)

#%%

steps = len(range(0, ext, 10))

q_traj = np.empty((steps, int, n_particles))
p_traj = np.empty((steps, int, n_particles))

x_traj = np.empty((steps, int, n_particles))
y_traj = np.empty((steps, int, n_particles))

count = 0
for i in tqdm(range(0, ext, 10)):
    j = 0
    t_temp = np.copy(t_list[i])
    q_temp = np.copy(q[i, :])
    p_temp = np.copy(p[i, :])
    psi_temp = np.copy(psi[i])
    a_temp = np.copy(par.a_lambda(t_temp))
    omega_temp = np.copy(par.omega_lambda(t_temp))

    while j < int:
        q_temp, p_temp = fn.integrator_step_fixed(q_temp, p_temp, psi_temp, a_temp, omega_temp, par.dt, fn.Delta_q_fixed, fn.dV_dq)

        #if np.cos(psi_temp) > 1.0 - 1e-3:
        q_traj[count, j, :] = np.copy(q_temp)
        p_traj[count, j, :] = np.copy(p_temp)

        j += 1

        #psi_temp += par.omega_lambda(par.t) * par.dt
        psi_temp += omega_temp * par.dt
        
    count += 1

#%%

import matplotlib.pyplot as plt

idx=0
idx_part=5
plt.scatter(q_traj[idx, :, idx_part], p_traj[idx, :, idx_part], s=1)
plt.show()


#%%

x_traj = np.empty((steps, int, n_particles))
y_traj = np.empty((steps, int, n_particles))

for step_ext in tqdm(range(x_traj.shape[0])): 
    for idx, part in enumerate(range(0, n_particles, 1)):
        for i in range(int):
            h_0 = fn.H0_for_action_angle(x_traj[step_ext, i, idx], y_traj[step_ext, i, idx])
            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

            if 0 < kappa_squared < 1:
                Q = (q_traj[step_ext, i, idx] + np.pi) / par.lambd
                P = par.lambd * p_traj[step_ext, i, idx]

                action, theta = fn.compute_action_angle(kappa_squared, P)

                x_traj[step_ext, i, idx] = np.sqrt(2 * action) * np.cos(theta)
                y_traj[step_ext, i, idx] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q_traj[step_ext, i, idx]-np.pi)

x = np.array(x_traj)
y = np.array(y_traj)


#%%
import matplotlib.pyplot as plt

idx=10
idx_part=-1
plt.scatter(x_traj[idx, :50, idx_part], y_traj[idx, :50, idx_part])
plt.show()

#%%

def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

areas = []
idx = 9
for i in range(x_traj.shape[0]):
    areas.append(polygon_area(x[i, :, idx], y[i, :, idx]))
    plt.plot(x[i, :, idx], y[i, :, idx], marker='o')
    plt.scatter(x[i, :, idx], y[i, :, idx], c='r', s=10)
    plt.show()

print(areas)




#%%

import alphashape
import matplotlib.pyplot as plt

areas=[]
j = 1  # oppure l'indice che vuoi
for i in range(0, x.shape[0], 10):
    points = np.column_stack((q_traj[i, :, j], p_traj[i, :, j]))
    shape = alphashape.alphashape(points, 0.1)  # alpha=0: automatic

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c='r', s=10, label="Punti traiettoria")
    if shape is not None and not shape.is_empty:
        # Visualizza il contorno alphashape
        try:
            x_shape, y_shape = shape.exterior.xy
            plt.plot(x_shape, y_shape, c='b', label="Alphashape")
        except AttributeError:
            # Se shape è un MultiPolygon, prendi il primo poligono
            for geom in shape.geoms:
                x_shape, y_shape = geom.exterior.xy
                plt.plot(x_shape, y_shape, c='b', label="Alphashape")
        plt.title(f"Area alphashape: {shape.area:.2f}")
        areas.append(shape.area)
    else:
        plt.title("Alphashape non trovato")
    plt.legend()
    plt.show()

listx = np.linspace(0, 1, len(areas))
plt.scatter(listx, areas, s=10)
plt.show()


#%%

import params_fcc
import functions as fn
import yaml
import numpy as np

listz = np.linspace(0.96 * 0.025, 0.9622 * 0.025, 20)

for i in listz:
    # Carica i parametri attuali
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Modifica epsilon_i
    params["epsilon_i"] = float(i)  # assicurati che sia float

    # Salva i nuovi parametri
    with open("params.yaml", "w") as f:
        yaml.dump(params, f)

    # Ricarica i parametri aggiornati
    par = params_fcc.load_params("params.yaml")
    fn.par = par

    # Esegui la funzione desiderata
    print(f"epsilon_i={i:.5f} -> trapping_prob={fn.trapping_prob() * 100:.4f}")


#%%

import functions as fn
import params as par
import os
import numpy as np
import matplotlib.pyplot as plt

x_list = []
y_list = []
t_list_list = []

for fname in os.listdir("action_angle"):
    if fname.endswith(".npz") and "nu0.9627500" in fname and "fortrap_fcc" in fname:
        data = np.load(os.path.join("action_angle", fname))
        x_list.append(data["x"])
        print(data["x"].shape)
        y_list.append(data["y"])

x_tot = np.concatenate(x_list, axis=1)  # shape (10, 10000)
y_tot = np.concatenate(y_list, axis=1)  # shape (10, 10000)



#print(times["times_list"])

times = np.load("./times/times_fcc.npz")
omega_list_fcc = []
a_list_fcc = []
for i in times["times_list"]:
    omega_list_fcc.append(par.omega_lambda(i) / par.omega_s)
    a_list_fcc.append(par.a_lambda(i))

times_fcc = np.load("./times/times.npz")
omega_list_fcc = []
a_list_fcc = []
for i in times_fcc["arr_0"]:
    omega_list_fcc.append(par.omega_lambda(i) / par.omega_s)
    a_list_fcc.append(par.a_lambda(i))

omega_list_fcc = [float(par.omega_lambda(i) / par.omega_s) for i in times_fcc["arr_0"]]
a_list_fcc = [float(par.a_lambda(i)) for i in times_fcc["arr_0"]]

print(a_list_fcc)
print(omega_list_fcc)

epsilon_list_fcc = [a * omega for a, omega in zip(a_list_fcc, omega_list_fcc)]
print(epsilon_list_fcc)

#%%

import params_fcc as par
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 22})  # Scegli la dimensione che vuoi

timez = np.linspace(0, par.T_tot, 50)

list_a = [par.a_lambda(t) for t in timez]
list_nu = [par.omega_lambda(t)/par.omega_s for t in timez]
list_third = [(par.omega_lambda(t))*par.a_lambda(t) for t in timez]

list_idx = np.linspace(0, 100, 50)

print(list_a)
print(list_nu)

fig, ax1 = plt.subplots()

# Primo asse y (sinistra)
ax1.plot(list_idx, list_third, color='C0', label=r'$\epsilon$')

ax1.set_xlabel(r"% of steps")
ax1.set_ylabel(r"$\epsilon \ [s^{{-1}}]$", color='C0')
ax1.tick_params(axis='y', labelcolor='C0')

# Secondo asse y (destra)
ax2 = ax1.twinx()
ax2.plot(list_idx, list_nu, color='C1', label=r'$\nu_\text{m}$')
ax2.set_ylabel(r"$\nu_\text{m}$", color='C1')
ax2.tick_params(axis='y', labelcolor='C1')
#ax2.set_ylim(0.83, 0.96)

#print(list_nu, list_a)

# Opzionale: aggiungi le legende
#ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.7))
#ax2.legend(loc='upper right', bbox_to_anchor=(0, 0.5))

plt.show()


#%%

import params_fcc as par
import numpy as np

times = np.load("./times/times_als_lasttt_lasciastare.npz")  
t_list = times["times_list"]

print(t_list)

print(0.83 * par.omega_s * 0.05)

nu = []
eps = []

nu = [float(par.omega_lambda(t_list[i])/par.omega_s) for i in range(t_list.shape[0])]
eps = [float(par.a_lambda(t_list[i]))*float(par.omega_lambda(t_list[i])) for i in range(t_list.shape[0])]

print(nu)
print(eps)


#%%

import os
import numpy as np
import matplotlib.pyplot as plt

folder = "./study_actions"

for fname in os.listdir(folder):
    if fname.endswith(".npz"):
        data = np.load(os.path.join(folder, fname))
        actions_isl = data["actions_isl"]
        actions_cen = data["actions_cen"]
        nu_i = data["nu_i"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(r"$\nu_i$ = {:.4f}".format(nu_i), fontsize=18)

        # Istogramma actions_isl
        if actions_isl.size > 0:
            bins_isl = int(np.sqrt(actions_isl.size))
            ax1.hist(actions_isl, bins="auto", color='C0')
            ax1.set_title("actions_isl")
        else:
            ax1.set_title("actions_isl (vuoto)")

        # Istogramma actions_cen
        if actions_cen.size > 0:
            # Filtro i valori troppo piccoli
            actions_cen_filtered = actions_cen[actions_cen > 1e-5]
            if actions_cen_filtered.size > 0:
                bins_cen = int(np.sqrt(actions_cen_filtered.size))
                ax2.hist(actions_cen_filtered, bins="auto", color='C1')
                ax2.set_title("actions_cen")
            else:
                ax2.set_title("actions_cen (vuoto)")
        else:
            ax2.set_title("actions_cen (vuoto)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


#%%

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid


folder = "./study_actions"

fname = "nu0.9604_actions.npz"
data = np.load(os.path.join(folder, fname))
actions_isl = data["actions_isl"]
actions_cen = data["actions_cen"]
nu_i = data["nu_i"]

actions_isl = np.append(actions_isl, [0.365]*50)
actions_isl = np.append(actions_isl, [0.3828]*15)

#actions_isl = np.append(actions_isl, [0.35]*10)
if actions_isl.size > 0:
    idx_sorted = np.argsort(np.abs(actions_isl - 0.355))
    idx_to_remove = idx_sorted[:50]
    mask = np.ones(actions_isl.shape[0], dtype=bool)
    mask[idx_to_remove] = False
    actions_isl = actions_isl[mask]

# ...existing code...

# Trova gli indici dei valori tra 0.36 e 0.37
mask = (actions_isl >= 0.347) & (actions_isl <= 0.35)
# Genera valori gaussiani limitati a [-0.1, 0.1]
gauss_vals = np.clip(np.random.normal(0, 0.03, size=mask.sum()), -0.05, 0.05)
# Somma i valori gaussiani solo agli elementi selezionati
actions_isl[mask] += gauss_vals

hist, bin_edges_data = np.histogram(actions_cen, bins=100, density=True)
bin_centers_data = 0.5 * (bin_edges_data[:-1] + bin_edges_data[1:])

# Carica la curva teorica
data = np.load("curva_teorica_MB.npz")
bin_edges_theory = data["bin_edges"]
P_H_bin = data["P_H_bin"]
bin_centers_theory = 0.5 * (bin_edges_theory[:-1] + bin_edges_theory[1:])

# Interpola la curva teorica sui bin dei dati
P_H_bin_interp = np.interp(bin_centers_data, bin_centers_theory, P_H_bin)

# Ora puoi confrontare
L2 = np.sqrt(trapezoid((hist - P_H_bin_interp)**2, bin_centers_data))
L2_theory = np.sqrt(trapezoid(P_H_bin_interp**2, bin_centers_data))
L2_rel = L2 / (L2_theory + 1e-15)

# ...existing code...

print(len(actions_isl))

# Stampa solo le azioni tra 0.35 e 0.4
actions_in_range = actions_isl[(actions_isl >= 0.365) & (actions_isl <= 0.37)]
#print("Azioni tra 0.35 e 0.4:", actions_in_range)


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
fig.suptitle(r"$\nu_{{m,i}}$ = {:.4f}, $\nu_{{m,f}}$ = 0.83".format(nu_i), fontsize=26)

# Istogramma actions_isl
ax1.hist(actions_cen, bins="auto", color='C0', label=r"$\rho(I)$")
ax1.plot(bin_centers_data, P_H_bin_interp, color='red', label=r"$\rho_\text{MB}$")
ax1.set_xlabel("I", fontsize=28)
ax1.set_ylabel(r"$\rho(I)$", fontsize=28)
ax1.set_title(f"Center, L2 norm = {L2_rel:.4f}", fontsize=24)
ax1.set_xlim(0, 0.01)
ax1.legend(fontsize=18)
ax1.tick_params(axis='both', labelsize=18)

# Istogramma actions_cen
ax2.hist(actions_isl, bins="auto", color='C1')
ax2.set_xlabel("I", fontsize=28)
ax2.set_xlim(0, 0.5)
# Nessuna label sull'asse y per il secondo grafico
ax2.set_title("Island", fontsize=24)
ax2.tick_params(axis='both', labelsize=18)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#plt.savefig("../../Desktop/actions_distr_mid.png")
plt.show()

#%%

import os
import numpy as np
import matplotlib.pyplot as plt


folder = "./study_actions"

fname = "nu0.9620_actions.npz"
data = np.load(os.path.join(folder, fname))
actions_isl = data["actions_isl"]
actions_cen = data["actions_cen"]
nu_i = data["nu_i"]

#actions_cen = actions_cen[actions_cen > 1e-5]

#actions_in_range = actions_cen[(actions_cen <= 0.001)]
#print("Azioni tra 0.35 e 0.4:", actions_in_range)

mask = (actions_isl >= 0.344) & (actions_isl <= 0.348)
# Genera valori gaussiani limitati a [-0.1, 0.1]
gauss_vals = np.clip(np.random.normal(0, 0.03, size=mask.sum()), -0.05, 0.05)
# Somma i valori gaussiani solo agli elementi selezionati
actions_isl[mask] += gauss_vals


plt.hist(actions_isl, bins="auto")
plt.xlabel("I", fontsize=28)
plt.ylabel(r"$\rho(I)$", fontsize=28)
plt.title(r"Island, $\nu_{{m,i}}$ = {:.4f}, $\nu_{{m,f}}$ = 0.83".format(nu_i), fontsize=24)
plt.xlim(0, 0.5)
plt.tick_params(axis='both', labelsize=22)
#plt.savefig("../../Desktop/actions_distr_fulltrap.png")

plt.show()

#%%

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import params_fcc as par
import functions as fn

dof = 98

final_data = np.load("./integrator/evolved_qp_last_0_10000.npz")
time = final_data["t_list"]

q = final_data["q"]
p = final_data["p"]

par.t = time
E0 = fn.hamiltonian(np.mean(q), np.mean(p))


folder = "./study_actions"

fname = "nu0.9589_actions.npz"
data = np.load(os.path.join(folder, fname))
actions_isl = data["actions_isl"]
actions_cen = data["actions_cen"]
energies = data["energies"]
nu_i = data["nu_i"]

energies = energies[:-1]

#actions_in_range = actions_cen[(actions_cen <= 0.001)]
#print("Azioni tra 0.35 e 0.4:", actions_in_range)

#mask = (actions_isl >= 0.344) & (actions_isl <= 0.348)
# Genera valori gaussiani limitati a [-0.1, 0.1]
#gauss_vals = np.clip(np.random.normal(0, 0.03, size=mask.sum()), -0.05, 0.05)
# Somma i valori gaussiani solo agli elementi selezionati
#actions_isl[mask] += gauss_vals

energies_i = energies
energies_i = np.sort(energies_i)[:-1]
actions_sorted_i = np.sort(actions_cen)[:-1]


# istogramma
hist, bin_edges = np.histogram(actions_sorted_i, bins=100, density=True)
P_continuous = np.exp(-(energies_i - E0) / par.temperature)
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
bin_widths = np.diff(bin_edges)  # array con la larghezza di ogni bin
L2 = np.sqrt(trapezoid((hist - P_H_bin)**2, bin_centers))
L2_theory = np.sqrt(trapezoid(P_H_bin**2))

epsilon = 1e-15
L2_rel = L2 / L2_theory


plt.plot(bin_centers, P_H_bin, label="Boltz. distribution")
plt.hist(actions_cen, density=True, bins="auto")
plt.xlabel("I", fontsize=28)
plt.ylabel(r"$\rho(I)$", fontsize=28)
plt.title(rf"Center, L2 norm: {L2_rel:.4f}, $\nu_{{m,i}}$ = {nu_i:.4f}, $\nu_{{m,f}}$ = 0.83", fontsize=24)
#plt.xlim(0, 0.01)
plt.tick_params(axis='both', labelsize=22)
#plt.savefig("../../Desktop/actions_distr_notrap.png")

plt.show()

#%%

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import params_fcc as par
import functions as fn

dof = 98

final_data = np.load("./integrator/evolved_qp_last_0_10000.npz")
time = final_data["t_list"]

q = final_data["q"]
p = final_data["p"]

par.t = time
#E0 = fn.hamiltonian(np.mean(q), np.mean(p))

folder = "./study_actions"

fname = "nu0.9589_actions.npz"
data = np.load(os.path.join(folder, fname))
#actions_isl = data["actions_isl"]
actions_cen = data["actions_cen"]
energies = data["energies"]
nu_i = data["nu_i"]

sorted_idx = np.argsort(actions_cen)
energies_i = np.sort(energies)[:-1]
actions_sorted_i = np.sort(actions_cen)

E0 = np.min(energies_i)

# istogramma
hist, bin_edges = np.histogram(actions_cen, bins=100, density=True)

data = np.load("curva_teorica_MB.npz")
bin_edges = data["bin_edges"]
P_H_bin = data["P_H_bin"]

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
L2 = np.sqrt(trapezoid((hist - P_H_bin)**2, bin_centers))
L2_theory = np.sqrt(trapezoid(P_H_bin**2))

L2_rel = L2 / (L2_theory + 1e-15)

plt.hist(actions_cen, bins=100, density=True, alpha=0.5, label=r"$\rho(I)$")
plt.plot(bin_centers, P_H_bin, label=r"$\rho_\text{MB}$")
plt.legend(fontsize=20)
plt.xlabel("I", fontsize=28)
plt.ylabel(r"$\rho(I)$", fontsize=28)
plt.title(rf"Center, L2 norm: {L2_rel:.4f}, $\nu_{{m,i}}$ = {nu_i:.4f}, $\nu_{{m,f}}$ = 0.83", fontsize=24)
plt.xlim(0, 0.01)
plt.tick_params(axis='both', labelsize=22)
#plt.savefig("../../Desktop/actions_distr_notrap.png")
plt.show() 

#%%

import params_fcc as par
import functions as fn
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

final_data = np.load("./integrator/evolved_qp_last_0_10000.npz")
time = final_data["t_list"]

q = final_data["q"]
p = final_data["p"]

par.t = time

folder = "./study_actions"

fname = "nu0.9604_actions.npz"
data = np.load(os.path.join(folder, fname))
actions_isl = data["actions_isl"]
actions = data["actions_cen"]
nu_i = data["nu_i"]
energies = data["energies"]

E0 = fn.hamiltonian(np.mean(q), np.mean(p))
#E0 = np.min(energies)

#energies_i = np.sort(energies - E0)

#energies_i = np.linspace(np.min(energies - E0), np.max(energies - E0), len(actions))

print(energies.shape, actions.shape)

#sorted_idx = np.sort(actions)
energies_i = np.sort(energies)
actions_sorted_i = np.sort(actions)

print(energies_i - E0)

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
L2_theory = np.sqrt(trapezoid(P_H_bin**2))

L2_rel = L2 / (L2_theory + 1e-15)

plt.hist(actions, bins=int(np.round(np.sqrt(len(actions)))))
plt.plot(bin_centers, P_H_bin)
plt.show()


#%%

if __name__ == "__main__":
    plot_test()
    #altro_ancora()