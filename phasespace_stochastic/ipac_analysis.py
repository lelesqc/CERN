import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import params_fcc_ww as params
from scipy.optimize import curve_fit

par = params.Params()
folder_type = "nu" 

if folder_type == "eps":
    folder_path = "./final_data_otherconfig/ww/eps_0.065"
    param_values = np.linspace(0.850, 0.900, 11)
    
    title_text = r'$\varepsilon = 0.065$'
    xlabel_text = r'$\nu_\mathrm{m}$'
    filename_out = "eq_emittance_fixed_eps.png"
    var_symbol = r'\nu_\mathrm{m}'

elif folder_type == "nu":
    folder_path = "./final_data_otherconfig/ww/nu_0.85"
    param_values = np.linspace(0.065, 0.075, 11) * par.omega_s
    
    title_text = r'$\nu_\mathrm{m} = 0.85$'
    xlabel_text = r'$\varepsilon$'
    filename_out = "eq_emittance_fixed_nu.png"
    var_symbol = r'\varepsilon'

emittance_list = []
file_list = sorted(glob.glob(os.path.join(folder_path, "*.npz")))

if len(file_list) != 11:
    print(f"Attenzione: Trovati {len(file_list)} file invece di 11 nella cartella {folder_path}.")

for file_path in file_list:
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']
    
    x_data = x if x.ndim == 1 else x[-1, :]
    y_data = y if y.ndim == 1 else y[-1, :]
    
    x0 = np.mean(x_data)
    y0 = np.mean(y_data)
    
    X = np.vstack([x_data - x0, y_data - y0])
    Sigma = np.cov(X)
    emittance = np.sqrt(np.linalg.det(Sigma))
    emittance_list.append(emittance)

def fit_powerlaw_asympt(x, A, B, p):
    return A + B * (x ** p)

def fit_powerlaw(x, B, p):
    return B * (x ** p)

x_data = np.array(param_values)
y_data = np.array(emittance_list)

popt_pl, pcov_pl = curve_fit(fit_powerlaw, x_data, y_data)
err_pl = np.sqrt(np.diag(pcov_pl))

popt_asym, pcov_asym = curve_fit(fit_powerlaw_asympt, x_data, y_data)
err_asym = np.sqrt(np.diag(pcov_asym))

A = popt_asym[0]
B = popt_asym[1]
p = popt_asym[2]

B_pl = popt_pl[0]
p_pl = popt_pl[1]

print(f"Fit power-law asymptotic: p = {p:.3f} +/- {err_asym[2]:.3f}, A = {A:.6f}, B = {B:.3f}")
print(f"Fit power-law: p = {p_pl:.3f} +/- {err_pl[1]:.3f}, B = {B_pl:.3f}")

x_line = np.linspace(min(x_data), max(x_data), 200)
y_fitted = fit_powerlaw_asympt(x_line, A, B, p)

plt.figure(figsize=(10, 5))

plt.scatter(x_data, y_data * 1000, color='blue', s=80, label="Simulation", zorder=3)

label_fit = rf"${A*1000:.2f} \cdot 10^{{-3}} + ({B*1000:.1f} \cdot 10^{{-3}} \ \text{{s}}^{{{p:.2f}}}) \cdot (a \omega_\text{{m}})^{{ {p:.2f} }}$"

plt.plot(x_line, y_fitted * 1000, color='blue', linewidth=2.5, label=label_fit, zorder=2)
plt.axhline(y=A * 1000, color='gray', lw=2.5, linestyle='--', alpha=1, zorder=1)
plt.xlabel(r"$a \omega_\text{m}$ [$\text{s}^{-1}$]", fontsize=20)
plt.ylabel(r"Equilibrium emittance $[10^{-3}]$", fontsize=20)

plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(2, 6)
plt.xlim(10, 41)

plt.grid(False)

plt.tight_layout()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
from scipy.stats import linregress, skew, kurtosis

config = "als"
os.environ["MACHINE"] = "ALS"
os.environ["CONFIG"] = config
os.environ["THERMAL_BATH"] = "yes"
os.environ["MODULATION"] = "yes"

if config != "als":
    os.environ["PARAMS_MODULE"] = f"params_fcc_{config}"
else:
    os.environ["PARAMS_MODULE"] = f"params_{config}"

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

nu = 0.94

#base_dir = "./ipac_simulations/a_0.03"
#list_nu = np.load(base_dir + "/nu_0.87/final_distr_cen.npz")
base_dir = f"./other_config/{config}/nu_{nu}/eps_0.028/"
list_nu = np.load(base_dir + "full_distr_cen.npz")
x_nu = list_nu["x"]
y_nu = list_nu["y"]

omega_m = nu * par.omega_s
T_s = 2 * np.pi / par.omega_s
dt = T_s / par.N
T_mod = 2 * np.pi / omega_m
steps = int(round(T_mod / dt))
n_steps = steps * par.N_turn

times = np.linspace(0, dt * n_steps, x_nu.shape[0])

moments_3 = {
    'x^3': [],  
         
    'y^3': []   
}

# Order 4
moments_4 = {
    'x^4': [],
    'x^2 y^2': [],  
    'y^4': []       
}

for i in range(times.shape[0]):
    x_raw = x_nu[i, :]
    y_raw = y_nu[i, :]
    
    x = x_raw - np.mean(x_raw)
    y = y_raw - np.mean(y_raw)
    
    sig_x = np.std(x)
    sig_y = np.std(y)
    
    if sig_x == 0: sig_x = 1e-9
    if sig_y == 0: sig_y = 1e-9
    
    moments_3['x^3'].append(   np.mean(x**3)        / sig_x**3 )
    #moments_3['x^2 y'].append( np.mean(x**2 * y**1) / (sig_x**2 * sig_y**1) )
    #moments_3['x y^2'].append( np.mean(x**1 * y**2) / (sig_x**1 * sig_y**2) )
    moments_3['y^3'].append(   np.mean(y**3)        / sig_y**3 )
    
    moments_4['x^4'].append(     np.mean(x**4)        / sig_x**4 )
    #moments_4['x^3 y'].append(   np.mean(x**3 * y**1) / (sig_x**3 * sig_y**1) )
    moments_4['x^2 y^2'].append( np.mean(x**2 * y**2) / (sig_x**2 * sig_y**2) )
    #moments_4['x y^3'].append(   np.mean(x**1 * y**3) / (sig_x**1 * sig_y**3) )
    moments_4['y^4'].append(     np.mean(y**4)        / sig_y**4 )

plt.figure(figsize=(10, 6))

plt.title(rf"Center, $\nu_m$ = {nu}", fontsize=14)

for label, data in moments_3.items():
    plt.plot(times, data, label=f"${label}$", lw=1.5, alpha=0.8)

plt.xlabel("Time [s]")
plt.ylabel("3rd Order Moments (Skewness)")
plt.axhline(0, color='k', linestyle='--', alpha=0.5, label='Gaussian (0)')
plt.legend(loc='upper right', ncol=4, fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.title(rf"Center, $\nu_m$ = {nu}", fontsize=14)

for label, data in moments_4.items():
    ls = '--' if ('x^3' in label or 'y^3' in label) else '-'
    plt.plot(times, data, label=f"${label}$", linestyle=ls, lw=1.5, alpha=0.8)

plt.xlabel("Time [s]")
plt.ylabel("4th Order Moments (Kurtosis)")

plt.axhline(3, color='gray', linestyle=':', alpha=0.5, label='Pure Gaussian (3)')
plt.axhline(1, color='gray', linestyle='-.', alpha=0.5, label='Mixed Gaussian (1)')

plt.legend(loc='upper right', ncol=5, fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import importlib

config = "als"
os.environ["MACHINE"] = "ALS"
os.environ["CONFIG"] = config
os.environ["THERMAL_BATH"] = "yes"
os.environ["MODULATION"] = "yes"
if config != "als":
    os.environ["PARAMS_MODULE"] = f"params_fcc_{config}"
else:
    os.environ["PARAMS_MODULE"] = f"params_{config}"

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

nu = 0.94

base_dir = f"./other_config/{config}/nu_{nu}/"
list_nu = np.load(base_dir + "eps_0.028/full_distr_cen.npz")
x_nu = list_nu["x"]
y_nu = list_nu["y"]

print(x_nu.shape[0])

omega_m = nu * par.omega_s
T_s = 2 * np.pi / par.omega_s
dt = T_s / par.N
T_mod = 2 * np.pi / omega_m
steps = int(round(T_mod / dt))
n_steps = steps * par.N_turn

times = np.linspace(0, dt * n_steps, x_nu.shape[0])

moments_sim = {'x^4': [], 'y^4': [], 'x^2 y^2': [], 'x^3 y': [], 'x y^3': []}
moments_theo = {'x^4': [], 'y^4': [], 'x^2 y^2': [], 'x^3 y': [], 'x y^3': []}

for i in range(times.shape[0]):
    x_raw = x_nu[i, :]
    y_raw = y_nu[i, :]
    
    x = x_raw - np.mean(x_raw)
    y = y_raw - np.mean(y_raw)
    sig_x = np.std(x)
    sig_y = np.std(y)
    
    if sig_x == 0: sig_x = 1e-9
    if sig_y == 0: sig_y = 1e-9

    rho = np.mean(x * y) / (sig_x * sig_y)
    
    moments_theo['x^4'].append(3.0)
    moments_theo['y^4'].append(3.0)
    moments_theo['x^2 y^2'].append(1.0 + 2.0 * rho**2)
    moments_theo['x^3 y'].append(3.0 * rho)
    moments_theo['x y^3'].append(3.0 * rho)
    
    moments_sim['x^4'].append(     np.mean(x**4)        / sig_x**4 )
    moments_sim['y^4'].append(     np.mean(y**4)        / sig_y**4 )
    moments_sim['x^2 y^2'].append( np.mean(x**2 * y**2) / (sig_x**2 * sig_y**2) )
    moments_sim['x^3 y'].append(   np.mean(x**3 * y**1) / (sig_x**3 * sig_y**1) )
    moments_sim['x y^3'].append(   np.mean(x**1 * y**3) / (sig_x**1 * sig_y**3) )

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title(r"Pure 4th Order Moments", fontsize=14)

c_x4 = 'tab:blue'
c_y4 = 'tab:purple'

ax1.plot(times, moments_sim['x^4'], color=c_x4, label=r'$\langle x^4 \rangle$', lw=1, alpha=0.9)
ax1.plot(times, moments_sim['y^4'], color=c_y4, label=r'$\langle y^4 \rangle$', lw=1, alpha=0.9)
ax1.axhline(3.0, color='gray', linestyle='--', lw=1, label=r'$E(x^4), E(y^4) = 3$')
ax1.set_xlabel("Time [s]", fontsize=12)
ax1.set_ylabel("Normalized Pure Moment", fontsize=12)
#ax1.set_ylim(2.8, 3.8) 
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper right', fontsize='medium', frameon=True)

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.set_title(r"Mixed 4th Order Moments", fontsize=14)

c_x2y2 = 'tab:green'
c_x3y  = 'tab:orange'
c_xy3  = 'tab:red'

ax2.plot(times, moments_sim['x^2 y^2'], color=c_x2y2, label=r'$\langle x^2 y^2 \rangle$', lw=1)
#ax2.plot(times, moments_sim['x^3 y'],   color=c_x3y,  label=r'$\langle x^3 y \rangle$',   lw=1)
#ax2.plot(times, moments_sim['x y^3'],   color=c_xy3,  label=r'$\langle x y^3 \rangle$',   lw=1)

ax2.plot(times, moments_theo['x^2 y^2'], color=c_x2y2, linestyle='--', lw=1, alpha=0.6)
#ax2.plot(times, moments_theo['x^3 y'],   color=c_x3y,  linestyle='--', lw=1, alpha=0.6)
#ax2.plot(times, moments_theo['x y^3'],   color=c_xy3,  linestyle='--', lw=1, alpha=0.6)

# Formatting
ax2.set_xlabel("Time [s]", fontsize=12)
ax2.set_ylabel("Normalized Mixed Moment", fontsize=12)
#ax2.set_ylim(2.0, 3.5) 
ax2.grid(True, linestyle='--', alpha=0.4)

legend_sim = ax2.legend(loc='upper right', fontsize='medium', frameon=True)
ax2.add_artist(legend_sim)

line_even = Line2D([0], [0], color=c_x2y2, linestyle='--', lw=1, label=r'$E[x^2 y^2] = 1 + 2\rho^2(t)$')
line_odd  = Line2D([0], [0], color=c_xy3,  linestyle='--', lw=1, label=r'$E[x^3 y], E[x y^3] = 3\rho(t)$')

ax2.legend(handles=[line_even, line_odd], loc='upper left', fontsize='medium', 
           frameon=True)

plt.tight_layout()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import importlib
from scipy.stats import skew, kurtosis

os.environ["MACHINE"] = "FCC"
os.environ["THERMAL_BATH"] = "yes"
os.environ["MODULATION"] = "yes"
os.environ["PARAMS_MODULE"] = "params_fcc"

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

base_dir = "./ipac_simulations/eps_0.028"
folder_nu = "nu_0.88"  # Il valore rappresentativo che abbiamo scelto

data_isl = np.load(os.path.join(base_dir, folder_nu, "final_distr_isl.npz"))
x_isl = data_isl["x"]

data_cen = np.load(os.path.join(base_dir, folder_nu, "final_distr_cen.npz"))
x_cen = data_cen["x"]

omega_m = float(folder_nu.split("_")[1]) * par.omega_s
T_s = 2 * np.pi / par.omega_s
dt = T_s / par.N
T_mod = 2 * np.pi / omega_m
steps = int(round(T_mod / dt))
n_steps = steps * par.N_turn

times = np.linspace(0, dt * n_steps, x_isl.shape[0])

skew_cen, skew_isl = [], []
kurt_cen, kurt_isl = [], []

for i in range(times.shape[0]):
    xi_cen = x_cen[i, :]
    xi_isl = x_isl[i, :]
    
    skew_cen.append(skew(xi_cen))
    skew_isl.append(skew(xi_isl))
    
    kurt_cen.append(kurtosis(xi_cen, fisher=True))
    kurt_isl.append(kurtosis(xi_isl, fisher=True))

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 5.5), sharex=True)

color_cen = 'gray'
color_isl = 'tab:blue'

ax1.plot(times, skew_isl, color=color_isl, label=r'Island', lw=2, alpha=0.9)
ax1.plot(times, skew_cen, color=color_cen, label=r'Centre', lw=2, alpha=0.9)
ax1.axhline(0.0, color='black', linestyle='--', lw=1.5, label='Gaussian profile')

ax1.set_ylabel(r"Skewness", fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, linestyle='--', alpha=0.4)

ax1.legend(loc='lower right', fontsize=12, frameon=True, ncol=3)

ax2.plot(times, kurt_isl, color=color_isl, lw=2, alpha=0.9)
ax2.plot(times, kurt_cen, color=color_cen, lw=2, alpha=0.9)
ax2.axhline(0.0, color='black', linestyle='--', lw=1.5)

ax2.set_xlabel("Time [s]", fontsize=14)
ax2.set_ylabel(r"Excess Kurtosis", fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.subplots_adjust(hspace=0.1)
plt.tight_layout()

plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import importlib

# --- CONFIGURATION ---
os.environ["MACHINE"] = "FCC"
os.environ["THERMAL_BATH"] = "yes"
os.environ["MODULATION"] = "yes"
os.environ["PARAMS_MODULE"] = "params_fcc"

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

base_dir = "./ipac_simulations/eps_0.028"
folder_nu = "nu_0.87"

data_isl = np.load(os.path.join(base_dir, folder_nu, "final_distr_isl.npz"))
x_isl, y_isl = data_isl["x"], data_isl["y"]

data_cen = np.load(os.path.join(base_dir, folder_nu, "final_distr_cen.npz"))
x_cen, y_cen = data_cen["x"], data_cen["y"]

omega_m = 0.87 * par.omega_s
T_s = 2 * np.pi / par.omega_s
dt = T_s / par.N
T_mod = 2 * np.pi / omega_m
steps = int(round(T_mod / dt))
n_steps = steps * par.N_turn

times = np.linspace(0, dt * n_steps, x_isl.shape[0])

delta_x3_cen, delta_y3_cen, delta_x2y_cen = [], [], []
delta_x4_cen, delta_y4_cen = [], []

delta_x3_isl, delta_y3_isl, delta_x2y_isl = [], [], []
delta_x4_isl, delta_y4_isl = [], []

for i in range(times.shape[0]):
    xc = x_cen[i, :] - np.mean(x_cen[i, :])
    yc = y_cen[i, :] - np.mean(y_cen[i, :])
    sx_c, sy_c = np.std(xc), np.std(yc)
    if sx_c == 0: sx_c = 1e-9
    if sy_c == 0: sy_c = 1e-9
    rho_c = np.mean(xc * yc) / (sx_c * sy_c)
    
    delta_x3_cen.append(np.mean(xc**3) / sx_c**3 - 0.0)
    delta_y3_cen.append(np.mean(yc**3) / sy_c**3 - 0.0)
    delta_x2y_cen.append(np.mean(xc**2 * yc) / (sx_c**2 * sy_c) - 0.0)
    delta_x4_cen.append(np.mean(xc**4) / sx_c**4 - 3.0)
    delta_y4_cen.append(np.mean(yc**4) / sy_c**4 - 3.0)

    xi = x_isl[i, :] - np.mean(x_isl[i, :])
    yi = y_isl[i, :] - np.mean(y_isl[i, :])
    sx_i, sy_i = np.std(xi), np.std(yi)
    if sx_i == 0: sx_i = 1e-9
    if sy_i == 0: sy_i = 1e-9
    rho_i = np.mean(xi * yi) / (sx_i * sy_i)
    
    delta_x3_isl.append(np.mean(xi**3) / sx_i**3 - 0.0)
    delta_y3_isl.append(np.mean(yi**3) / sy_i**3 - 0.0)
    delta_x2y_isl.append(np.mean(xi**2 * yi) / (sx_i**2 * sy_i) - 0.0)
    delta_x4_isl.append(np.mean(xi**4) / sx_i**4 - 3.0)
    delta_y4_isl.append(np.mean(yi**4) / sy_i**4 - 3.0)

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)

c_isl = 'blue'
c_cen = 'gray'

ls_x = '-'      # Solid for X
ls_y = '--'     # Dashed for Y
ls_xy = '-.'    # Dash-dot for mixed X^2 Y^2

ax1.plot(times, delta_x3_isl, color=c_isl, linestyle=ls_x, lw=2, alpha=0.7)
ax1.plot(times, delta_y3_isl, color=c_isl, linestyle=ls_y, lw=2, alpha=0.7)
ax1.plot(times, delta_x3_cen, color=c_cen, linestyle=ls_x, lw=2, alpha=0.7)
ax1.plot(times, delta_y3_cen, color=c_cen, linestyle=ls_y, lw=2, alpha=0.7)
#ax1.plot(times, delta_x2y_isl, color="tab:blue", linestyle=ls_xy, lw=2, alpha=0.7)
#ax1.plot(times, delta_x2y_cen, color="dimgrey", linestyle=ls_xy, lw=2, alpha=0.7)

ax1.axhline(0.0, color='black', linestyle=':', lw=2)
ax1.set_ylabel(r"$\Delta$ Skewness", fontsize=18)
ax1.set_ylim(-1, 1)
ax1.grid(False)

ax2.plot(times, delta_x4_isl, color=c_isl, linestyle=ls_x, lw=2, alpha=0.7)
ax2.plot(times, delta_y4_isl, color=c_isl, linestyle=ls_y, lw=2, alpha=0.7)
ax2.plot(times, delta_x4_cen, color=c_cen, linestyle=ls_x, lw=2, alpha=0.7)
ax2.plot(times, delta_y4_cen, color=c_cen, linestyle=ls_y, lw=2, alpha=0.7)

ax2.axhline(0.0, color='black', linestyle=':', lw=2)
ax2.set_xlabel("Time [s]", fontsize=18)
ax2.set_ylabel(r"$\Delta$ Kurtosis", fontsize=18)
ax2.set_ylim(-1, 1)
ax2.grid(False)

leg_style = [
    Line2D([0], [0], color='black', linestyle=ls_x, lw=2, label=r'$X$'),
    Line2D([0], [0], color='black', linestyle=ls_y, lw=2, label=r'$Y$'),
]
leg_color = [
    Line2D([0], [0], color='blue', lw=2),
    Line2D([0], [0], color=c_cen, lw=2)
]

labels_color = ['Island', 'Centre']

l1 = ax1.legend(handles=leg_style, loc='lower right', fontsize=14, frameon=True, ncols=3)

from matplotlib.legend_handler import HandlerTuple

l2 = ax2.legend(handles=leg_color, labels=labels_color, loc='lower right', 
                fontsize=14, frameon=True, ncols=2)
plt.subplots_adjust(hspace=0.025)
ax1.yaxis.get_major_ticks()[0].label1.set_visible(False) 

plt.show()


# %%


import numpy as np
import os

data_isl = np.load("./ipac_simulations/eps_0.028/nu_0.87/final_distr_isl.npz")
x = data_isl["x"][-1, :] - np.mean(data_isl["x"][-1, :])
y = data_isl["y"][-1, :] - np.mean(data_isl["y"][-1, :])

sx = np.std(x)
sy = np.std(y)
rho = np.mean(x * y) / (sx * sy)

print(f"Correlation rho: {rho:.4f}\n")

def check_moment(name, value, theory):
    diff = value - theory
    print(f"{name:15} | Sim: {value:8.4f} | Theory: {theory:8.4f} | Delta: {diff:8.4f}")
    return diff

# --- 3rd order ---
check_moment("<x^2 y>", np.mean(x**2 * y) / (sx**2 * sy), 0.0)
check_moment("<x y^2>", np.mean(x * y**2) / (sx * sy**2), 0.0)

print("-" * 60)

# --- 4th order ---
check_moment("<x^3 y>", np.mean(x**3 * y) / (sx**3 * sy), 3 * rho)
check_moment("<x y^3>", np.mean(x * y**3) / (sx * sy**3), 3 * rho)
check_moment("<x^2 y^2>", np.mean(x**2 * y**2) / (sx**2 * sy**2), 1 + 2*rho**2)

print("-" * 60)

check_moment("<x^3 y^2>", np.mean(x**3 * y**2) / (sx**3 * sy**2), 0.0)

# %%

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import params_fcc_ww as params
from scipy.optimize import curve_fit

par = params.Params()

folder_type = "nu"  # Options: "eps", "nu"

if folder_type == "eps":
    folder_path = "./final_data_otherconfig/ww/eps_0.065"
    param_values = np.linspace(0.850, 0.900, 11)
    
    title_text = r'$\varepsilon = 0.065$'
    xlabel_text = r'$\nu_\mathrm{m}$'
    filename_out = "eq_emittance_fixed_eps.png"
    var_symbol = r'\nu_\mathrm{m}'

elif folder_type == "nu":
    folder_path = "./final_data_otherconfig/ww/nu_0.85"
    param_values = np.linspace(0.060, 0.075, 16) * par.omega_s
    
    title_text = r'$\nu_\mathrm{m} = 0.85$'
    xlabel_text = r'$\varepsilon$'
    filename_out = "eq_emittance_fixed_nu.png"
    var_symbol = r'\varepsilon'

else:
    raise ValueError("'eps' or 'nu'")

emittance_list = []
file_list = sorted(glob.glob(os.path.join(folder_path, "*.npz")))

for file_path in file_list:
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']
    
    x_data = x if x.ndim == 1 else x[-1, :]
    y_data = y if y.ndim == 1 else y[-1, :]
    
    x0 = np.mean(x_data)
    y0 = np.mean(y_data)
    
    X = np.vstack([x_data - x0, y_data - y0])
    Sigma = np.cov(X)
    emittance = np.sqrt(np.linalg.det(Sigma))
    emittance_list.append(emittance)

emittance_array = np.array(emittance_list)

xs = np.asarray(param_values)          
ys = emittance_array * 1e3             
xn = xs.mean()                         

print("N file:", len(file_list), " N param:", len(xs))
for fn, xv, yv in zip(file_list, xs, ys):
    print(f"{os.path.basename(fn):40s}  x={xv:6.2f}  y={yv:.4f}")

print("\nProfilo su y0:")
for y0_fix in [1.0, 1.5, 1.8, 2.0, 2.06, 2.15]:
    f = lambda x, B, p: y0_fix + B * (x / xn) ** p
    po, _ = curve_fit(f, xs, ys, p0=[ys.mean() - y0_fix, -1], maxfev=20000)
    rms = np.sqrt(np.mean((ys - f(xs, *po)) ** 2))
    print(f"y0={y0_fix:.2f}  B={po[0]:.3f}  p={po[1]:.2f}  RMS={rms:.4f}")

def pl_off(x, y0, B, p):  return y0 + B * (x / xn) ** p
def exp_off(x, y0, B, lam): return y0 + B * np.exp(-(x - xn) / lam)

po, pc = curve_fit(pl_off, xs, ys, p0=[2.0, 0.5, -2], maxfev=20000)
print("\npower-law + offset:", np.round(po, 3), "err:", np.round(np.sqrt(np.diag(pc)), 3),
      "RMS:", round(np.sqrt(np.mean((ys - pl_off(xs, *po)) ** 2)), 4))

po, pc = curve_fit(exp_off, xs, ys, p0=[2.0, 0.5, 10], maxfev=20000)
print("exp + offset:      ", np.round(po, 3), "err:", np.round(np.sqrt(np.diag(pc)), 3),
      "RMS:", round(np.sqrt(np.mean((ys - exp_off(xs, *po)) ** 2)), 4))

# --- Linear/Power fit ---

x_fit = np.linspace(param_values.min(), param_values.max(), 200)

if folder_type == "eps":
    y_scaled = emittance_array * 1e3 
    m, q = np.polyfit(param_values, emittance_array, 1)
    
    fit_line = m * param_values + q
    y_plot_fit = fit_line * 1e3
    
    m_scaled = m * 1e3
    q_scaled = q * 1e3
    sign_q = "+" if q_scaled >= 0 else "-"
    eq_label = fr'${m_scaled:.1f} \cdot 10^{{-3}} \cdot \nu_\mathrm{{m}} {sign_q} {abs(q_scaled):.1f} \cdot 10^{{-3}}$'
    
    x_plot_fit = param_values
    y_plot_scatter = y_scaled
    y0_line = None # Nessun asintoto per il caso lineare

elif folder_type == "nu":
    def fit_powerlaw_asympt(x, y0, A, b):
        return y0 + A * (x ** b)

    p0 = [0.0021, 0.1, -1.7] 
    
    popt, _ = curve_fit(fit_powerlaw_asympt, param_values, emittance_array, p0=p0, maxfev=10000)
    y0_fit, A_fit, b_fit = popt

    print(y0_fit, A_fit, b_fit)
    
    x_plot_fit = x_fit
    y_plot_fit_raw = fit_powerlaw_asympt(x_fit, y0_fit, A_fit, b_fit)
    
    y_plot_scatter = emittance_array * 1000
    y_plot_fit = y_plot_fit_raw * 1000
    y0_line = y0_fit * 1000
    A_scaled = A_fit * 1000
    A_str = f"{A_scaled:.1e}".replace("e+", r" \cdot 10^{")
    eq_label = rf"${y0_fit*1000:.2f} \cdot 10^{{-3}} + ({A_str}}} \cdot 10^{{-3}} \ \text{{s}}^{{{b_fit:.2f}}}) \cdot (a \omega_\text{{m}})^{{ {b_fit:.2f} }}$"

# --- 3. Creazione Grafico ---
plt.figure(figsize=(10, 5))

if folder_type == "nu":
    plt.axhline(y=y0_line, color='gray', linestyle='--', linewidth=2.5, alpha=1, zorder=1)

plt.scatter(param_values, y_plot_scatter, color='blue', s=80, label='Simulation', zorder=3)
plt.plot(x_plot_fit, y_plot_fit, color='blue', linewidth=2.5, label=eq_label, zorder=2)

if folder_type == "nu":
    plt.xlabel(r"$a \omega_\text{m}$ [$\text{s}^{-1}$]", fontsize=20)
else:
    plt.xlabel(xlabel_text, fontsize=20)

plt.ylabel(r"Equilibrium emittance $[10^{-3}]$", fontsize=20)
plt.title(title_text, fontsize=20)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.legend(fontsize=18)
plt.grid(False)
plt.tight_layout()

# output_dir = "./../../Desktop/plots/FCC_ee_WW"
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, filename_out)
# plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()


# %%
