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
    param_values = np.linspace(0.065, 0.075, 16) * par.omega_s
    
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
import importlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

config_data = "z"       # "z", "ww", "als"
folder_type = "nu"      # "eps": scansione di nu_m
                         # "nu":  scansione di epsilon

params_module = (
    "params_als" if config_data == "als"
    else f"params_fcc_{config_data}"
)

Params = importlib.import_module(params_module).Params
par = Params()

#base_dir = f"./final_data_otherconfig/{config_data}"
base_dir = f"./new_attempts/{config_data}"

# --------------------------------------------------
# Individuazione automatica dei dati
# --------------------------------------------------

if folder_type == "eps":
    folder_prefix = "eps_"
    file_prefix = "isl_nu_"

elif folder_type == "nu":
    folder_prefix = "nu_"
    file_prefix = "isl_eps_"

else:
    raise ValueError("folder_type deve essere 'eps' oppure 'nu'")

folders = sorted(glob.glob(
    os.path.join(base_dir, f"{folder_prefix}*")
))

# Considera soltanto cartelle contenenti file validi
folders = [
    folder for folder in folders
    if os.path.isdir(folder)
    and glob.glob(os.path.join(folder, f"{file_prefix}*.npz"))
]

if len(folders) != 1:
    raise ValueError(
        f"Trovate {len(folders)} cartelle compatibili:\n"
        + "\n".join(folders)
        + "\nSeleziona esplicitamente la cartella desiderata."
    )

folder_path = folders[0]

# Parametro mantenuto costante
fixed_value = float(
    os.path.basename(folder_path).removeprefix(folder_prefix)
)

# File della scansione
file_list = glob.glob(
    os.path.join(folder_path, f"{file_prefix}*.npz")
)


# Estrae i valori dai nomi dei file e li ordina numericamente
pairs = sorted(
    (
        float(
            os.path.basename(file)
            .removeprefix(file_prefix)
            .removesuffix(".npz")
        ),
        file
    )
    for file in file_list
)

# Mantiene l'ordine corretto dei file
file_list = [file for _, file in pairs]

# Ricostruisce i parametri mediante linspace
param_values = np.linspace(
    pairs[0][0],
    pairs[-1][0],
    len(file_list)
)


# --------------------------------------------------
# Conversione e titoli
# --------------------------------------------------

if folder_type == "eps":
    # Scansione di nu_m a epsilon fisica costante
    epsilon_fixed = fixed_value * par.omega_s

    xlabel = r"$\nu_{\mathrm{m}}$"
    title = (
        rf"$a\omega_\mathrm{{m}} = "
        rf"{epsilon_fixed:.1f}\,\mathrm{{s}}^{{-1}}$"
    )
    filename_out = "eq_emittance_fixed_eps.png"

else:
    # Scansione di epsilon fisica a nu_m costante
    nu_fixed = fixed_value
    param_values *= par.omega_s

    xlabel = r"$a\omega_{\mathrm{m}}$ [$\mathrm{s}^{-1}$]"
    title = rf"$\nu_\mathrm{{m}} = {nu_fixed:.2f}$"
    filename_out = "eq_emittance_fixed_nu.png"

print(f"Configurazione: {config_data}")
print(f"Cartella: {folder_path}")
print(f"Numero di punti: {len(file_list)}")
print(f"Intervallo: {param_values.min():.5f} - {param_values.max():.5f}")

if len(file_list) != len(param_values):
    raise ValueError(
        f"Trovati {len(file_list)} file, ma sono attesi "
        f"{len(param_values)} punti."
    )

emittance_list = []

for file_path in file_list:
    with np.load(file_path) as data:
        x = data["x"]
        y = data["y"]

    x_data = x if x.ndim == 1 else x[-1, :]
    y_data = y if y.ndim == 1 else y[-1, :]

    X = np.vstack([
        x_data - np.mean(x_data),
        y_data - np.mean(y_data)
    ])

    covariance = np.cov(X)
    emittance = np.sqrt(np.linalg.det(covariance))
    emittance_list.append(emittance)

ys = np.asarray(emittance_list) * 1e3
xs = np.asarray(param_values)

# --------------------------------------------------
# Fit
# --------------------------------------------------

if folder_type == "eps":

    # Legge lineare: eps_eq = m * nu_m + q
    # Fit lineare con matrice di covarianza
    coeffs, pcov = np.polyfit(xs, ys, 1, cov=True)

    m, q = coeffs

    # Errori standard sui parametri
    m_err, q_err = np.sqrt(np.diag(pcov))

    print("\nFit lineare:")
    print(f"m = {m:.6f} +/- {m_err:.6f}")
    print(f"q = {q:.6f} +/- {q_err:.6f}")

    # Verifica preliminare dell'ipotesi m = -q
    sum_mq = m + q
    sum_mq_err = np.sqrt(
        pcov[0, 0] + pcov[1, 1] + 2 * pcov[0, 1]
    )

    print(f"m + q = {sum_mq:.6f} +/- {sum_mq_err:.6f}")

    x_plot = np.linspace(xs.min(), xs.max(), 200)
    y_plot = m * x_plot + q

    print(f"\nFit lineare: eps_eq = m * nu_m + q")
    print(f"m = {m:.6g}")
    print(f"q = {q:.6g}")

    equation_label = (
        rf"${m:.1f}\,\nu_{{\mathrm{{m}}}} {q:+.1f}$"
    )

else:

    # Legge di potenza con offset:
    # eps_eq = y0 + A * x**b

    def power_law(x, y0, A, b):
        return y0 + A * x**b

    p0 = [
        max(0.0, ys.min() * 0.8),
        max(ys.max() - ys.min(), 1e-6),
        -1.0
    ]

    bounds = (
        [0.0, 0.0, -15.0],
        [ys.min(), np.inf, -0.1]
    )

    popt, pcov = curve_fit(
        power_law,
        xs,
        ys,
        p0=p0,
        bounds=bounds,
        maxfev=50000
    )


    x_plot = np.linspace(xs.min(), xs.max(), 200)
    y_plot = power_law(x_plot, *popt)

    y0_fit, A_fit, b_fit = popt
    y0_err, A_err, b_err = np.sqrt(np.diag(pcov))

    print("\nFit a legge di potenza con offset:")
    print(f"A = {y0_fit:.6f} +/- {y0_err:.6f}")
    print(f"B = {A_fit:.6e} +/- {A_err:.6e}")
    print(f"p = {b_fit:.6f} +/- {b_err:.6f}")

    A_str = f"{A_fit:.1e}"
    mantissa, exponent = A_str.split("e")
    exponent = int(exponent)

    equation_label = (
        rf"${y0_fit:.2f} + "
        rf"({mantissa} \cdot 10^{{{exponent}}} "
        rf"\,\mathrm{{s}}^{{{b_fit:.2f}}})"
        rf"\cdot (a\omega_\mathrm{{m}})^{{{b_fit:.2f}}}$"
    )

# --------------------------------------------------
# Grafico
# --------------------------------------------------

# --- 3. Creazione Grafico ---
plt.figure(figsize=(10, 5))

if folder_type == "nu":
    plt.axhline(
        y=y0_fit,
        color="gray",
        linestyle="--",
        linewidth=2.5,
        alpha=1,
        zorder=1
    )
    plt.xlabel(r"$a\omega_\text{m}$ [$\text{s}^{-1}$]", fontsize=20)
elif folder_type == "eps":
    plt.xlabel(r"$\nu_\text{m}$", fontsize=20)


plt.scatter(
    xs, ys,
    color="blue",
    s=80,
    label="Simulation",
    zorder=3
)

plt.plot(
    x_plot, y_plot,
    color="blue",
    linewidth=2.5,
    label=equation_label,
    zorder=2
)

plt.ylabel(r"Equilibrium emittance $[10^{-3}]$", fontsize=20)
plt.title(title, fontsize=20)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.legend(fontsize=18)
plt.grid(False)
plt.tight_layout()

output_dir = f"./../../Desktop/plots/{'ALS' if config_data == 'als' else f'FCC_ee_{config_data.upper()}'}"
os.makedirs(output_dir, exist_ok=True)

#plt.savefig(os.path.join(output_dir, filename_out), dpi=300, bbox_inches="tight")

plt.show()


# %% --- 2. Scansione Chi^2 vs Esponente b ---


# ============================================================
# COMPARISON: OLD VS NEW EQUILIBRIUM EMITTANCE
# ============================================================

old_dir = f"./final_data_otherconfig/{config_data}/nu_0.850"
new_dir = f"./new_attempts/{config_data}/nu_0.850"


def load_emittances(directory):
    files = glob.glob(
        os.path.join(directory, "isl_eps_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No simulation files found in {directory}"
        )

    results = []

    for path in files:
        # Modulation parameter u = a * nu_m
        u = float(
            os.path.basename(path)
            .removeprefix("isl_eps_")
            .removesuffix(".npz")
        )

        with np.load(path) as data:
            x = data["x"]
            y = data["y"]

        x_data = x if x.ndim == 1 else x[-1, :]
        y_data = y if y.ndim == 1 else y[-1, :]

        X = np.vstack([
            x_data - np.mean(x_data),
            y_data - np.mean(y_data)
        ])

        emittance = np.sqrt(
            max(0.0, np.linalg.det(np.cov(X)))
        )

        results.append((u, emittance * 1e3))

    results.sort(key=lambda item: item[0])
    return np.asarray(results)


old_data = load_emittances(old_dir)
new_data = load_emittances(new_dir)

u_old, em_old = old_data.T
u_new, em_new = new_data.T

# Only compare the overlapping modulation interval
u_min = max(u_old.min(), u_new.min())
u_max = min(u_old.max(), u_new.max())

mask_old = (u_old >= u_min) & (u_old <= u_max)
mask_new = (u_new >= u_min) & (u_new <= u_max)

u_old, em_old = u_old[mask_old], em_old[mask_old]
u_new, em_new = u_new[mask_new], em_new[mask_new]

print(f"\nConfiguration: {config_data.upper()}")
print(f"Common interval: [{u_min:.5f}, {u_max:.5f}]")
print(f"Old points: {len(u_old)}")
print(f"New points: {len(u_new)}")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.scatter(
    u_old, em_old,
    label="Old simulations",
    marker="o",
    s=65
)

plt.scatter(
    u_new, em_new,
    label="New simulations",
    marker="x",
    s=75
)

plt.xlabel(r"$u=a\nu_m$", fontsize=16)
plt.ylabel(
    r"Equilibrium emittance $[10^{-3}]$",
    fontsize=16
)
plt.title(
    f"{config_data.upper()} - Old vs new simulations",
    fontsize=17
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Numerical comparison at old simulation points
# ------------------------------------------------------------


# Compare only old points within the new dataset range
mask = (
    (u_old >= u_new.min()) &
    (u_old <= u_new.max())
)

u_compare = u_old[mask]
em_old_compare = em_old[mask]

# Interpolate using the FULL new dataset
em_new_interp = np.interp(
    u_compare, u_new, em_new
)

relative_diff = 100 * (
    em_new_interp - em_old_compare
) / em_old_compare

print("\nComparison (new interpolated at old u):")
print("    u       Old       New       Difference (%)")

for u, old, new, diff in zip(
    u_compare,
    em_old_compare,
    em_new_interp,
    relative_diff
):
    print(
        f"{u:8.5f} "
        f"{old:9.5f} "
        f"{new:9.5f} "
        f"{diff:+10.3f}"
    )

print(
    "\nMean absolute relative difference: "
    f"{np.mean(np.abs(relative_diff)):.3f}%"
)



#%%

from scipy.stats import f
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# RSS profile and fixed-exponent analysis
# --------------------------------------------------

# Fixed exponents selected for the paper
fixed_exponents = {
    "z": -1.75,
    "ww": -4.5,
    "als": -1.5
}

config = config_data.lower()
b_target = fixed_exponents[config]

# Exponent range
b_min = min(b_fit, b_target) - 0.5
b_max = max(b_fit, b_target) + 0.5
b_vals = np.linspace(b_min, b_max, 500)

# Reference modulation strength for numerical stability
x0 = np.sqrt(xs.min() * xs.max())


def fit_fixed_exponent(b):
    z = (xs / x0)**b

    X = np.column_stack([
        np.ones_like(z),
        z
    ])

    coeffs, _, _, _ = np.linalg.lstsq(
        X, ys, rcond=None
    )

    A, C = coeffs
    B = C / (x0**b)

    residuals = ys - X @ coeffs
    rss = np.sum(residuals**2)

    return np.array([A, B]), rss


# RSS profile
rss_vals = np.array([
    fit_fixed_exponent(b)[1] for b in b_vals
])

# Fit with fixed exponent
coeffs_target, rss_target = fit_fixed_exponent(b_target)

# Original free-exponent fit
rss_free = np.sum(
    (ys - power_law(xs, *popt))**2
)

# Relative increase in RSS
rss_increase = 100 * (rss_target / rss_free - 1)

# Nested-model F-test
n = len(xs)

F_stat = (rss_target - rss_free) / (
    rss_free / (n - 3)
)

p_value = f.sf(max(F_stat, 0), 1, n - 3)

# --------------------------------------------------
# Results
# --------------------------------------------------

print(f"\nConfiguration: {config.upper()}")
print(f"Free exponent: p = {b_fit:.5f}")
print(f"Fixed exponent: p = {b_target:.5f}")

print(f"RSS (free): {rss_free:.6e}")
print(f"RSS (fixed): {rss_target:.6e}")
print(f"RSS increase: {rss_increase:.3f}%")

print(f"F statistic: {F_stat:.4f}")
print(f"p-value: {p_value:.6f}")

print("\nFixed-exponent coefficients:")
print(f"A = {coeffs_target[0]:.6f}")
print(f"B = {coeffs_target[1]:.6e}")

# --------------------------------------------------
# Normalised RSS profile
# --------------------------------------------------

rss_normalized = rss_vals / rss_free

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    b_vals,
    rss_normalized,
    color="blue",
    linewidth=2.5
)

ax.axvline(
    b_fit,
    color="red",
    linestyle="--",
    linewidth=2,
    label=fr"Free fit: $p={b_fit:.3f}$"
)

ax.axvline(
    b_target,
    color="green",
    linestyle=":",
    linewidth=2,
    label=fr"Fixed exponent: $p={b_target:g}$"
)

ax.set_xlabel(r"Exponent $p$", fontsize=18)
ax.set_ylabel(
    r"$\mathrm{RSS}(p)/\mathrm{RSS}_{\mathrm{free}}$",
    fontsize=17
)

ax.tick_params(axis="both", labelsize=14)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=13)

fig.tight_layout()

fig.savefig(
    f"rss_profile_{config}.pdf",
    bbox_inches="tight"
)

#fig.savefig(f"../../Desktop/rss_profile_{config}.pdf", bbox_inches="tight")

plt.show()


# %%


# ============================================================
# CONFRONTO FIT LINEARE LIBERO E VINCOLATO (m = -q)
# ============================================================

from scipy.stats import f

if folder_type == "eps":

    # --------------------------------------------------------
    # 1. Fit libero: y = m*nu_m + q
    # --------------------------------------------------------

    coeff_free, cov_free = np.polyfit(xs, ys, 1, cov=True)

    m_free, q_free = coeff_free
    m_err, q_err = np.sqrt(np.diag(cov_free))

    y_free = m_free * xs + q_free
    rss_free = np.sum((ys - y_free)**2)

    # --------------------------------------------------------
    # 2. Fit vincolato: y = C*(1 - nu_m)
    # --------------------------------------------------------

    def linear_constrained(nu, C):
        return C * (1.0 - nu)

    popt_con, pcov_con = curve_fit(
        linear_constrained,
        xs,
        ys
    )

    C = popt_con[0]
    C_err = np.sqrt(pcov_con[0, 0])

    m_con = -C
    q_con = C

    y_con = linear_constrained(xs, C)
    rss_con = np.sum((ys - y_con)**2)

    # --------------------------------------------------------
    # 3. Confronto statistico
    # --------------------------------------------------------

    n = len(xs)

    # Fit libero: 2 parametri
    # Fit vincolato: 1 parametro

    F_stat = ((rss_con - rss_free) / 1) / (rss_free / (n - 2))

    p_value = f.sf(max(F_stat, 0.0), 1, n - 2)

    # --------------------------------------------------------
    # 4. Risultati
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print(f"CONFIGURAZIONE: {config_data.upper()}")
    print("CONFRONTO FIT LINEARE LIBERO / VINCOLATO")
    print("=" * 55)

    print("\nFIT LIBERO:")
    print(f"m = {m_free:.6f} +/- {m_err:.6f}")
    print(f"q = {q_free:.6f} +/- {q_err:.6f}")
    print(f"RSS = {rss_free:.6e}")

    print("\nFIT VINCOLATO (m = -q):")
    print(f"C = {C:.6f} +/- {C_err:.6f}")
    print(f"m = {-C:.6f}")
    print(f"q = {C:.6f}")
    print(f"RSS = {rss_con:.6e}")

    print("\nCONFRONTO:")
    print(f"Rapporto RSS = {rss_con/rss_free:.6f}")
    print(f"Incremento RSS = {(rss_con/rss_free - 1)*100:.3f}%")
    print(f"F = {F_stat:.6f}")
    print(f"p-value = {p_value:.6f}")

    # --------------------------------------------------------
    # 5. Grafico dei due fit
    # --------------------------------------------------------

    x_plot_comp = np.linspace(xs.min(), xs.max(), 300)

    plt.figure(figsize=(9, 5))

    plt.scatter(
        xs, ys,
        color="blue",
        s=70,
        label="Simulation",
        zorder=3
    )

    plt.plot(
        x_plot_comp,
        m_free * x_plot_comp + q_free,
        color="blue",
        linewidth=2.5,
        label="Unconstrained fit"
    )

    plt.plot(
        x_plot_comp,
        linear_constrained(x_plot_comp, C),
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=r"Constrained fit ($m=-q$)"
    )

    plt.xlabel(r"$\nu_m$", fontsize=18)
    plt.ylabel(r"Equilibrium emittance $[10^{-3}]$", fontsize=18)
    plt.title(f"{config_data.upper()} - Linear fit comparison")

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

else:
    print("Questo confronto richiede folder_type = 'eps'.")

#%%


import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import f as f_dist


# ============================================================
# CONFIGURATION
# ============================================================

# Select the current machine configuration:
# "z", "ww", or "als"
config = config_data.lower()

fixed_exponents = {
    "z": -1.75,
    "ww": -4.5,
    "als": -1.5,
}

# xs: physical modulation strength epsilon [s^-1]
# ys: equilibrium emittance / 1e-3
#
# Both arrays must already be defined by ipac_analysis.py.

xs = np.asarray(xs, dtype=float)
ys = np.asarray(ys, dtype=float)

if xs.ndim != 1 or ys.ndim != 1 or len(xs) != len(ys):
    raise ValueError("xs and ys must be 1D arrays of equal length.")

if np.any(xs <= 0):
    raise ValueError("Power-law fits require xs > 0.")

if len(xs) <= 3:
    raise ValueError("At least four points are required.")

p_fixed = fixed_exponents[config]
n = len(xs)


# ============================================================
# 1. UNCONSTRAINED POWER-LAW FIT
# ============================================================

def power_law(x, A, B, p):
    return A + B * x**p


# Same initialisation and bounds used in the original analysis.
p0 = [
    max(0.0, ys.min() * 0.8),
    max(ys.max() - ys.min(), 1e-6),
    -1.0,
]

bounds = (
    [0.0, 0.0, -15.0],
    [ys.min(), np.inf, -0.1],
)

popt, pcov = curve_fit(
    power_law,
    xs,
    ys,
    p0=p0,
    bounds=bounds,
    maxfev=100000,
)

A_free, B_free, p_free = popt
errors_free = np.sqrt(np.diag(pcov))

residuals_free = ys - power_law(xs, *popt)
rss_free = np.sum(residuals_free**2)


# ============================================================
# 2. PARAMETER CORRELATION MATRIX
# ============================================================

correlation = pcov / np.outer(errors_free, errors_free)

print("\n" + "=" * 60)
print(f"CONFIGURATION: {config.upper()}")
print("=" * 60)

print("\nFREE POWER-LAW FIT")
for name, value, error in zip(
    ["A", "B", "p"], popt, errors_free
):
    print(f"{name:>2} = {value:.8g} +/- {error:.4g}")

print(f"RSS_free = {rss_free:.10g}")

print("\nCORRELATION MATRIX (A, B, p)")
print(np.array2string(correlation, precision=5))

print(f"\nCorrelation B-p = {correlation[1, 2]:+.6f}")


# ============================================================
# 3. FIXED-EXPONENT FIT
# ============================================================


def fit_at_fixed_p(p):
    x0 = np.sqrt(xs.min() * xs.max())

    z = (xs / x0)**p
    X = np.column_stack([np.ones_like(z), z])

    coefficients, _, _, _ = np.linalg.lstsq(
        X, ys, rcond=None
    )

    A, C = coefficients
    B = C / (x0**p)

    residuals = ys - X @ coefficients
    rss = np.sum(residuals**2)

    variance = rss / (len(xs) - 2)

    cov_AC = variance * np.linalg.inv(X.T @ X)

    # Transform covariance from (A, C) to (A, B)
    J = np.diag([1.0, 1.0 / (x0**p)])
    covariance = J @ cov_AC @ J.T

    errors = np.sqrt(np.diag(covariance))

    return A, B, rss, covariance, errors



A_fixed, B_fixed, rss_fixed, cov_fixed, err_fixed = (
    fit_at_fixed_p(p_fixed)
)

rss_increase = 100 * (rss_fixed / rss_free - 1)

# Nested-model F test:
# free model: 3 parameters
# fixed-p model: 2 parameters
F_stat = max(0.0, (rss_fixed - rss_free)) / (
    rss_free / (n - 3)
)

p_value = f_dist.sf(F_stat, 1, n - 3)

print(f"\nFIXED EXPONENT: p = {p_fixed}")
print(f"A = {A_fixed:.8g} +/- {err_fixed[0]:.4g}")
print(f"B = {B_fixed:.8g} +/- {err_fixed[1]:.4g}")
print(f"RSS_fixed = {rss_fixed:.10g}")
print(f"RSS increase = {rss_increase:.3f}%")
print(f"F statistic = {F_stat:.5f}")
print(f"p-value = {p_value:.6f}")


# ============================================================
# 4. PROFILE RSS(p)
# ============================================================

# Profile around the free-fit exponent.
# Increase the interval if the profile is too narrow.
p_grid = np.linspace(
    p_free - 0.8,
    p_free + 0.8,
    500,
)

rss_profile = np.array([
    fit_at_fixed_p(p)[2] for p in p_grid
])

rss_normalized = rss_profile / rss_free

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    p_grid,
    rss_normalized,
    lw=2,
    label=r"$\mathrm{RSS}(p)/\mathrm{RSS}_{\mathrm{free}}$",
)

ax.axvline(
    p_free,
    color="black",
    linestyle="--",
    label=fr"Free fit: $p={p_free:.3f}$",
)

ax.axvline(
    p_fixed,
    color="tab:red",
    linestyle=":",
    label=fr"Fixed exponent: $p={p_fixed}$",
)

ax.scatter(
    [p_fixed],
    [rss_fixed / rss_free],
    color="tab:red",
    zorder=5,
)

ax.set_xlabel(r"Exponent $p$")
ax.set_ylabel(
    r"$\mathrm{RSS}(p)/\mathrm{RSS}_{\mathrm{free}}$"
)

ax.set_title(f"{config.upper()}: exponent sensitivity")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(
    f"rss_profile_{config}.pdf",
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 5. COMPARISON OF FREE AND FIXED FITS
# ============================================================

x_plot = np.linspace(xs.min(), xs.max(), 400)

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.scatter(
    xs,
    ys,
    color="black",
    s=25,
    label="Simulation data",
)

ax.plot(
    x_plot,
    power_law(x_plot, *popt),
    lw=2,
    label="Free exponent",
)

ax.plot(
    x_plot,
    power_law(x_plot, A_fixed, B_fixed, p_fixed),
    lw=2,
    linestyle="--",
    label=fr"Fixed exponent: $p={p_fixed}$",
)

ax.set_xlabel(r"Modulation strength $\varepsilon$ [s$^{-1}$]")
ax.set_ylabel(r"$\varepsilon_{\mathrm{eq}}/10^{-3}$")

ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(
    f"power_fit_comparison_{config}.pdf",
    bbox_inches="tight",
)

plt.show()

# %%
