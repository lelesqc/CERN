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

param_names = ["eta", "h", "radius", "gamma"]
params = [par_fcc.eta, par_fcc.h, par_fcc.radius, par_fcc.gamma]
energies_tot = []
energies_tot_temp = []

for name, default_val in zip(param_names, params):
    if name == "h":
        par_vals = np.linspace(par_als.h, default_val, 10)
        par_vals = np.round(par_vals).astype(int)
    elif name == "eta":
        par_vals = np.linspace(default_val, par_als.eta, 10)
    elif name == "radius":
        par_vals = np.linspace(par_als.radius, default_val, 10)
    elif name == "gamma":
        par_vals = np.linspace(par_als.gamma, default_val, 10)

    temps_th = []
    temps_emp = []

    for val in par_vals:
        setattr(par_fcc, name, val)
        base_dir="/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code"

        init_data = np.load("./init_conditions/qp_10000.npz")

        q = init_data["q"]
        p = init_data["p"]

        psi = 0
        par_fcc.t = 0

        n_particles = int(q.shape[0])

        q_sec = np.zeros(n_particles)
        p_sec = np.zeros(n_particles)
        sec_count = 0

        times = []
        step = 0
        for step in tqdm(range(par_fcc.n_steps)): 
            q, p = fn.integrator_step(q, p, psi, par_fcc.t, par_fcc.dt, fn.Delta_q, fn.dV_dq)

            if np.cos(psi) > 1.0 - 1e-3:              
                q_sec = q
                p_sec = p

            psi += par_fcc.omega_m * par_fcc.dt
            par_fcc.t += par_fcc.dt

        q = q_sec
        p = p_sec

        # shape di q e p: (n_punti, n_particelle)

        energies = []
        
        h_0 = fn.H0_for_action_angle(q, p)
        energies = h_0

        E0 = fn.hamiltonian(np.pi, 0)
        noise_D = (par_fcc.gamma / par_fcc.beta**2 * np.sqrt(2 * par_fcc.damp_rate * par_fcc.h * par_fcc.eta * par_fcc.Cq / par_fcc.radius))**2
        damping_factor = 2 * par_fcc.damp_rate / par_fcc.beta**2
        temperature = noise_D / (2 * damping_factor)
        temps_th.append(temperature)
        temps_emp.append(np.mean(energies - E0))

        energies_tot_temp.append(energies)

    energies_tot.append(energies_tot_temp)

    print(temps_th, temps_emp)
    plt.scatter(par_vals, temps_th)
    plt.show()
    plt.scatter(par_vals, temps_emp)
    plt.show()

    output_dir = os.path.join("stochastic_studies", "temperatures")
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, f"temp_{name}_mix.npz"), temps_th=temps_th, temps_emp=temps_emp, energies=energies_tot)

#%%

from scipy.stats import linregress
import functions as fn
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

dir_data = "./stochastic_studies/temperatures"
temps_dict = {}

for root, dirs, files in os.walk(dir_data):
    for file in files:
        if file.startswith("temp_") and file.endswith("_mix.npz"):
            param_name = file[len("temp_"):-len("_mix.npz")]  
            data = np.load(os.path.join(root, file))
            if "temps_th" in data and "temps_emp" in data:
                temps_dict[param_name] = {
                    "temps_th": data["temps_th"],
                    "temps_emp": data["temps_emp"]
                }

all_th = []
all_emp = []

for param_name, temps in temps_dict.items():
    if param_name == "radius":
        continue
    temps_th = np.array(temps["temps_th"])
    temps_emp = np.array(temps["temps_emp"])
    all_th.append(temps_th)
    all_emp.append(temps_emp)

all_th = np.concatenate(all_th)
all_emp = np.concatenate(all_emp)

a = np.array2string(np.max(all_th), formatter={'float_kind':lambda x: "%.2e" % x})
b = np.array2string(np.max(all_emp), formatter={'float_kind':lambda x: "%.2e" % x})

# Fit 1: solo coefficiente angolare (y = m*x)
m, = np.linalg.lstsq(all_th.reshape(-1,1), all_emp, rcond=None)[0]

#all_emp = all_emp - 3.851

# Fit 2: coefficiente angolare + intercetta (y = m*x + q)
m2, q2 = np.polyfit(all_th, all_emp, 1)


print(f"Fit y = m*x: m = {m}")
print(f"Fit y = m*x + q: m = {m2}, q = {q2}")

# Plot
param_names = list(temps_dict.keys())
cmap = plt.get_cmap('tab10')  # Colormap con 10 colori distinti

plt.figure(figsize=(8,6))
for i, param_name in enumerate(param_names):
    if param_name == "radius":
        continue
    temps_th = np.array(temps_dict[param_name]["temps_th"])
    temps_emp = np.array(temps_dict[param_name]["temps_emp"])
    plt.scatter(temps_th, temps_emp, color=cmap(i), label=param_name, s=14, alpha=1.0)

# Rette del fit
x_fit = np.linspace(np.min(all_th), np.max(all_th), 100)
plt.plot(x_fit, m * x_fit, '-', c="grey", label='Fit: y = m·x')
#plt.plot(x_fit, m2 * x_fit + q2, 'r-', label='Fit: y = m·x + q')

plt.xlabel("T theoretical")
plt.ylabel("T empirical")
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.title(r"Linear fit $T_{emp} \ vs \ T_{th}$ for FCC-ee")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

# %%
