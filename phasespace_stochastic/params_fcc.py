import numpy as np
import yaml

# ------------ machine ----------------

h = 130000 
eta = 14.8e-6 
nu_s = 0.025
T_rev = 326.08e-6
V = 0.1e9
E_s = 45.6e9
radius = 10.76e3 
U_0 = 0.039e9
omega_rev = 2 * np.pi / T_rev
k_B = 8.617333262e-5

# -------------- model -----------------

k_lele_fcc = 38689.07

damp_rate = U_0 / T_rev / E_s    # alpha_E
beta = 1.0
gamma = 8.92e4
N = 100    # fixed
N_turn = 100    # per avere T_tot = 3 * damping_time
phi_0 = 0.0
e = 1
lambd = np.sqrt(h * eta * omega_rev)

omega_s = omega_rev * np.sqrt(e * h * V * eta / (2 * np.pi * E_s * beta**2))
A = omega_s / lambd
Cq = 3.83 * 10e-13

# -------------- YAML ------------------

config_path = "params.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)

epsilon = config["epsilon"]
nu_m = config["nu_m"]

# ------------- variables -----------------

omega_m = nu_m * omega_s
a = epsilon / nu_m
T_s = 2 * np.pi / omega_s
dt = T_s / N
T_mod = 2 * np.pi / omega_m
steps = int(round(T_mod / dt))
n_steps = steps * N_turn

t = 0.0

#damp_rate = 0
a = 0

# tempo totale da usare: 3 * damp_factor = 0.19s 
print(damp_rate)