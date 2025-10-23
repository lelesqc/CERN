#%%

"""
QUESTO CODICE SERVE A CALCOLARE LA TEMPERATURA USANDO LA FORMULA
E USANDO LA MEDIA DELLE ENERGIE, IN MODO DA CONFRONTARLE E TROVARE
LA COSTANTE DI PROPORZIONALITà TRA LE DUE

USARE DIRETTAMENTE LA SECONDA CELLA

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from scipy.integrate import trapezoid

import params as par_als
import params_fcc as par_fcc
import functions as fn

import warnings
warnings.filterwarnings("ignore")

parameters = par_als
par_type = "als" if parameters is par_als else "fcc"

par = parameters.Params()

#param_names = ["h", "gamma", "radius", "omega_rev"]
#params = [par.h, par.gamma, par.radius, par.omega_rev]
param_names = ["h", "gamma", "omega_rev"]
params = [par.h, par.gamma, par.omega_rev]
temps_dict = {}
all_th = []
all_emp = []
all_emp_std = []

for name, default_val in zip(param_names, params):
    par = parameters.Params()
    
    par_vals = np.linspace(default_val / 10, default_val * 10, 5)
    par_vals = np.append(par_vals, default_val)
    par_vals = np.sort(par_vals)

    if name == "h":
        par_vals = np.round(par_vals).astype(int)

    print(par_vals)

    temps_th = []
    temps_emp = []
    temps_emp_std = []
    energies_tot_temp = []

    for val in par_vals:
        setattr(par, name, val)
        par.update_dependent()

        print(par.h, par.gamma, par.radius, par.omega_rev)
        #print(par.h, par.omega_rev")

        init_data = np.load("./init_conditions/qp_10000.npz")

        q = init_data["q"][::10]
        p = init_data["p"][::10]

        psi = 0
        par.t = 0

        n_particles = int(q.shape[0])

        q_sec = np.zeros(n_particles)
        p_sec = np.zeros(n_particles)
        sec_count = 0

        times = []
        step = 0
        for step in tqdm(range(par.n_steps)): 
            q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq, par)

            if np.cos(psi) > 1.0 - 1e-3:              
                q_sec = q
                p_sec = p

            psi += par.omega_m * par.dt
            par.t += par.dt

        q = q_sec
        p = p_sec

        # shape di q e p: (n_punti, n_particelle)
        
        h_0 = fn.hamiltonian(q, p, par)
        energies = h_0

        E0 = fn.hamiltonian(np.pi, 0, par)
        damp_factor = 2 * par.damp_rate / par.beta**2
        m = 1 / (par.h * par.eta * par.omega_rev)
        temperature = par.gamma**2 * par.h * par.eta * par.omega_rev * par.Cq / (2 * (2 + par.damping_part_number) * par.beta**4 * par.radius)  
        #D = par.gamma / par.beta**3 * np.sqrt(par.damp_rate * par.Cq / par.radius)
        temps_th.append(temperature)
        temps_emp.append(np.mean(energies - E0))
        #temps_emp.append(np.var(p) / (2 * m))
        temps_emp_std.append(np.std(energies - E0))
        energies_tot_temp.append(energies)

    temps_dict[name] = {
        "temps_th": np.array(temps_th),
        "temps_emp": np.array(temps_emp),
        "temps_emp_std": np.array(temps_emp_std)
    }

    # Aggiungi ai dati globali
    all_th.append(temps_th)
    all_emp.append(temps_emp)
    all_emp_std.append(temps_emp_std)

    plt.scatter(par_vals, temps_th)
    plt.show()
    plt.scatter(par_vals, temps_emp)
    plt.show()

    output_dir = os.path.join("stochastic_studies", "temperatures")
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, f"temp_{name}_{par_type}.npz"), temps_th=temps_th, temps_emp=temps_emp, temps_emp_std=temps_emp_std)

all_th = np.concatenate(all_th)
all_emp = np.concatenate(all_emp)
all_emp_std = np.concatenate(all_emp_std)

#%%

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker
import params as par

dir_data = "./stochastic_studies/temperatures"
par_type = "als"

temps_dict = {}
all_th = []
all_emp = []
all_emp_std = []

for root, dirs, files in os.walk(dir_data):
    for file in files:
        if file.startswith("temp_") and file.endswith(f"_{par_type}.npz"):
            param_name = file[len("temp_"):-len(f"_{par_type}.npz")]
            data = np.load(os.path.join(root, file))
            
            temps_th = np.array(data["temps_th"])
            temps_emp = np.array(data["temps_emp"])
            #temps_emp_std = np.array(data["temps_emp_std"])

            #if param_name in ["gamma"]:
            #    continue

            temps_dict[param_name] = {
                "temps_th": data["temps_th"],
                "temps_emp": data["temps_emp"],
                #"temps_emp_std": data["temps_emp_std"]
            }

            all_th.append(temps_th)
            all_emp.append(temps_emp)
            #all_emp_std.append(temps_emp_std)

all_th = np.concatenate(all_th)
all_emp = np.concatenate(all_emp)
#all_emp_std = np.concatenate(all_emp_std)

print(all_th.shape)

m, = np.linalg.lstsq(all_th.reshape(-1,1), all_emp, rcond=None)[0]    # y = m*x
m2, q2 = np.polyfit(all_th, all_emp, 1)    # y = m2*x + q2

print(f"Fit y = m*x: m = {m}")
print(f"Fit y = m*x + q: m = {m2}, q = {q2}")

# Plot
param_names = list(temps_dict.keys())
cmap = plt.get_cmap('tab10')

plt.figure(figsize=(8,6))
for i, param_name in enumerate(param_names):
    temps_th = np.array(temps_dict[param_name]["temps_th"])
    temps_emp = np.array(temps_dict[param_name]["temps_emp"])
    #temps_emp_std = np.array(temps_dict[param_name]["temps_emp_std"])
    #if param_name in ["gamma"]:
    #    continue

    #plt.errorbar(temps_th, temps_emp, yerr=temps_emp_std / np.sqrt(n_particles), fmt='o', color=cmap(i), markersize=2, capsize=4, alpha=1)
    plt.scatter(temps_th, temps_emp, color=cmap(i), label=param_name, s=14, alpha=1.0)

x_fit = np.linspace(np.min(all_th), np.max(all_th), 100)
plt.plot(x_fit, m * x_fit, '-', c="grey", label=f'Fit: y = m·x (m={m:.3g})')
#plt.plot(x_fit, m2 * x_fit + q2, '--', c="red", label=f'Fit: y = m·x + q (m={m2:.3g}, q={q2:.2g})')

plt.xlabel("T theoretical")
plt.ylabel("T empirical")
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.title(r"Linear fit $T_{emp} \ vs \ T_{th}$ for FCC-ee")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

for idx in range(10):
    print(f"{all_th[idx]:.3f}, {all_emp[idx]:.3f}")

# %%

