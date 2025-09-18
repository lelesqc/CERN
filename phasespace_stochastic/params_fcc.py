import numpy as np
import yaml

# ------------ machine ----------------

h = 130
eta = 14.8e-6
nu_s = 0.0075
T_rev = 0.66e-6
V = 1.5e6
E_s = 1.5e9
radius = 4.01
U_0 = 0.11e6
omega_rev = 2 * np.pi / T_rev

# -------------- model -----------------

damp_rate = U_0 / T_rev / E_s
beta = 1.0
gamma = 2935.42
N = 100    # fixed
N_turn = 1000
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

print(n)

#damp_rate = 0
a = 0

print(omega_s / lambd**2)

noise_factor = gamma / beta**2 * np.sqrt(2 * damp_rate * h * eta * Cq / radius)
print(noise_factor)
