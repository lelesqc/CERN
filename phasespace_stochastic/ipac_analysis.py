import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
from scipy.stats import linregress


params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

machine = os.environ.get("MACHINE").lower()

base_dir = "./ipac_simulations/a_0.03"
folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

list_nu = np.linspace(0.84, 0.94, 11)
phasespace = np.load(f"action_angle/phasespace_75_a0.0335714_nu0.84000_fcc.npz")
x_ps = phasespace["x"]
y_ps = phasespace["y"]

emittance_list = []

plt.figure(figsize=(8, 6))

for folder, nu in zip(folders, list_nu):
    npz_path = os.path.join(base_dir, folder, "final_distr_isl.npz")
    data = np.load(npz_path)
    
    x = data["x"]
    y = data["y"]
    
    x0 = np.mean(x[-1, :])
    y0 = np.mean(y[-1, :])

    X = np.vstack([x[-1, :] - x0, y[-1, :] - y0])  # shape (2, N)
    Sigma = np.cov(X)                       # (2, 2)

    X_points = X.T                          # (N, 2)
    det_Sigma = np.linalg.det(Sigma)
    emittance = np.sqrt(det_Sigma)
    emittance_list.append(emittance)

plt.scatter(list_nu, emittance_list)
plt.xlabel(r"$\nu_m$", fontsize=18)
plt.ylabel(r"$\varepsilon$", fontsize=18)
plt.title(r"Island, Final emittance vs. $\nu_m$")
plt.show()

data_nu = np.load(base_dir + "/nu_0.87/final_distr_isl.npz")
x_nu = data_nu["x"]
y_nu = data_nu["y"]

step = 1
instant_emit = []
indices = np.arange(0, x_nu.shape[0], step)

for i in indices:
    x_mean = np.mean(x_nu[i, :])
    y_mean = np.mean(y_nu[i, :])

    X = np.vstack([x_nu[i, :] - x_mean, y_nu[i, :] - y_mean])
    Sigma = np.cov(X)
    det_Sigma = np.linalg.det(Sigma)
    emit = np.sqrt(det_Sigma)
    instant_emit.append(emit)

x = np.linspace(0, par.dt * par.n_steps, indices.shape[0])
y = np.log(instant_emit)

linear_thr = 0.28
mask = x < linear_thr
x_fit = x[mask]
y_fit = y[mask]

res = linregress(x_fit, y_fit)
a, b = res.intercept, res.slope
r2 = res.rvalue**2
tau = (-1.0 / b) if b < 0 else np.inf
sigma_b = res.stderr
if b != 0:
    sigma_tau = abs(sigma_b / b**2)
else:
    sigma_tau = np.nan

print(f"Fit, tau={tau:.3f} s, damping time: {1 / (2 * par.damp_rate)}, damping rate: {par.damp_rate}, U0: {par.U_0}")

x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
L2_fit = np.exp(a + b * x_line)

#plt.plot(x_line, L2_fit, 'r-', lw=2, label=rf"$\tau_\text{{conv}}$=({tau:.2f} $\pm$ {sigma_tau:.2f})s, $\tau_{{damp}}$ = {1 / (2 * par.damp_rate):.2f} s")
plt.scatter(x, instant_emit, s=7)
plt.xlabel("Time [s]")
plt.ylabel(r"$\varepsilon$")
plt.title(r"Island, $\nu_m$ = 0.87, Emittance vs. Time")
plt.show()

"""import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

step = 2
indices = np.arange(0, x_nu.shape[0], step)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter([], [])
ax.set_xlim(np.min(x_nu), np.max(x_nu))
ax.set_ylim(np.min(y_nu), np.max(y_nu))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Evoluzione fasi x vs y")

def update(frame):
    i = indices[frame]
    sc.set_offsets(np.c_[x_nu[i, :], y_nu[i, :]])
    ax.set_title(f"Turn {i}")
    return sc,

ani = FuncAnimation(fig, update, frames=len(indices), blit=True, repeat=False)
plt.show()"""
    