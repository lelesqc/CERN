import numpy as np
import yaml

# ------------ machine ----------------

h = 328
eta = 1.26e-3
nu_s = 0.0075
omega_rev = 9_571_303.971
omega_s = 71784.77978
V = 1.5e6
E_s = 1.5e9

# -------------- model -----------------

fft_steps = None

damp_rate = 38.1
beta = 1.0
D = 1.82e-3
N = 100    # fixed
phi_0 = 0.0
e = 1
lambd = np.sqrt(h * eta * omega_rev)
A = omega_s / lambd

# -------------- YAML ------------------

config_path = "params.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)

epsilon = config["epsilon"]
nu_m = config["nu_m"]

# ------------- variables -----------------

omega_m = nu_m * omega_s
a = epsilon / nu_m
#a = 0
T_s = 2 * np.pi / omega_s
dt = T_s / N
T_mod = 2 * np.pi / omega_m
steps = int(round(T_mod / dt))

if fft_steps is None:
    N_turn = 2500
else:
    N_turn = np.ceil(fft_steps / N)

n_steps = steps * N_turn

t = 0.0 