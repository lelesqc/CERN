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

import params as par
import functions as fn

import warnings
warnings.filterwarnings("ignore")

param_names = ["Cq", "eta", "h", "radius", "gamma"]
params = [par.Cq, par.eta, par.h, par.radius, par.gamma]
energies_tot = []
energies_tot_temp = []

for name, default_val in zip(param_names, params):
    if name == "h":
        par_vals = np.linspace(default_val / 10, default_val * 10, 20)
        par_vals = np.round(par_vals).astype(int)
    else:
        par_vals = np.linspace(default_val / 10, default_val * 10, 20)

    temps_th = []
    temps_emp = []

    print(par_vals[:3], par.beta)

    for val in par_vals:
        setattr(par, name, val)
        base_dir="/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code"

        init_data = np.load("./init_conditions/qp_1000.npz")

        q = init_data["q"]
        p = init_data["p"]

        psi = 0
        par.t = 0

        n_particles = int(q.shape[0])

        q_sec = np.zeros(n_particles)
        p_sec = np.zeros(n_particles)
        sec_count = 0

        times = []
        step = 0
        for step in tqdm(range(par.n_steps)): 
            q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

            if np.cos(psi) > 1.0 - 1e-3:              
                q_sec = q
                p_sec = p

            psi += par.omega_m * par.dt
            par.t += par.dt

        q = q_sec
        p = p_sec

        # shape di q e p: (n_punti, n_particelle)

        #n_times = 10
        #step = int(len(times)/n_times)

        #actions = np.zeros((n_times, n_particles))
        energies = []
        #kappa_squared_list = []

        #for i, idx in enumerate(range(0, len(times), step)):
        h_0 = fn.H0_for_action_angle(q, p)
        energies = h_0
        #kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
        #actions[i, :], _ = fn.compute_action_angle(kappa_squared, 1)
        #h0_of_I = 2 * par.A**2 * (kappa_squared - 1/2)
            #energies.append(h0_of_I)

        #actions = np.array(actions)

        E0 = fn.hamiltonian(np.pi, 0)
        #E_min = np.min(energies[9])
        noise_D = (par.gamma / par.beta**2 * np.sqrt(2 * par.damp_rate * par.h * par.eta * par.Cq / par.radius))**2
        damping_factor = 2 * par.damp_rate / par.beta**2
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

    # Salva temperature teorica ed empirica in un file .npz
    output_dir = os.path.join("stochastic_studies", "temperatures")
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, f"temp_{name}_fcc.npz"), temps_th=temps_th, temps_emp=temps_emp, energies=energies_tot)

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
        if file.startswith("temp_") and file.endswith("_fcc.npz"):
            param_name = file[len("temp_"):-len("_fcc.npz")]  
            data = np.load(os.path.join(root, file))
            if "temps_th" in data and "temps_emp" in data:
                temps_dict[param_name] = {
                    "temps_th": data["temps_th"],
                    "temps_emp": data["temps_emp"]
                }

all_th = []
all_emp = []

for param_name, temps in temps_dict.items():
    temps_th = np.array(temps["temps_th"])
    temps_emp = np.array(temps["temps_emp"])
    mask = temps_th <= 0.08e2
    all_th.append(temps_th[mask])
    all_emp.append(temps_emp[mask])

# Unisci tutti i dati filtrati
all_th = np.concatenate(all_th)[:26]
all_emp = np.concatenate(all_emp)[:26]

a = np.array2string(np.max(all_th), formatter={'float_kind':lambda x: "%.2e" % x})
b = np.array2string(np.max(all_emp), formatter={'float_kind':lambda x: "%.2e" % x})
# Fit 1: solo coefficiente angolare (y = m*x)
m, = np.linalg.lstsq(all_th.reshape(-1,1), all_emp, rcond=None)[0]

# Fit 2: coefficiente angolare + intercetta (y = m*x + q)
m2, q2 = np.polyfit(all_th, all_emp, 1)

print(f"Fit y = m*x: m = {m}")
#print(f"Fit y = m*x + q: m = {m2}, q = {q2}")

# Plot
plt.scatter(all_th, all_emp, s=5, alpha=0.7)
x_fit = np.linspace(all_th.min(), all_th.max(), 100)
plt.plot(x_fit, m*x_fit, color="orange", label=f"Fit y=mx, m={m:.3g}")
idx = 29
plt.scatter(all_th, all_emp, s = 30)
#plt.plot(x_fit, m2*x_fit + q2, label=f"Fit y=mx+q, m={m2:.3g}, q={q2:.3g}")
plt.xlabel("T theoretical")
plt.ylabel("T empirical")
plt.legend()
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.xscale("log")
#plt.yscale("log")
plt.title(r"Linear fit $T_{th} \ vs \ T_{emp}$ for FCC-ee")

plt.show()

# %%
