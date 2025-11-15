import numpy as np
import yaml
import os     

# ------------ machine ----------------

bending_radius = 14430 
h = 130000
C_gamma = 8.85e-5    # m * GeV^-3
nu_s = 0.025
T_rev = 302.54e-6
V = 0.1e9
radius = 10.76e3
mc2 = 0.511e6
gamma = 89236.8
momentum_compaction = 14.8e-6 
damping_part_number = momentum_compaction * bending_radius / radius
E_s = gamma * mc2
#eta = 14.8e-6
eta = momentum_compaction - 1/gamma**2
U_0 = 0.039e9
#U_0 = C_gamma * (1e-9)**3 * E_s**4 / radius

omega_rev = 2 * np.pi / T_rev

# -------------- model -----------------

damp_rate = U_0 / (2 * T_rev * E_s) * (2 + damping_part_number)
beta = np.sqrt(1 - 1/gamma**2)
N = 100
N_turn = 2500  # circa 20 volte il damping time
phi_0 = 0.0
e = 1
lambd = np.sqrt(h * eta * omega_rev)
omega_s = omega_rev * np.sqrt(e * h * V * eta / (2 * np.pi * E_s * beta**2))
A = omega_s / lambd
Cq = 3.83e-13
#D = gamma / beta**3 * np.sqrt(damp_rate * Cq / radius)
#D = 0
damp_rate = 0
temperature = gamma**2 * h * eta * omega_rev * Cq / (2 * (2 + damping_part_number) * beta**4 * radius) 

# -------------- YAML ------------------

config_path = "params.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)

epsilon_i = config["epsilon_i"]
epsilon_f = config["epsilon_f"]
nu_m_i = config["nu_m_i"]
nu_m_f = config["nu_m_f"]

omega_m_i = nu_m_i * omega_s
omega_m_f = nu_m_f * omega_s

Delta_eps = epsilon_f - epsilon_i
Delta_omega = (nu_m_f - nu_m_i) * omega_s

# ------------- variables -----------------

b_target = 0.005
f1 = 0.1
f2 = 0.9

T_s = 2 * np.pi / omega_s
dt = T_s / N
T_mod = 2 * np.pi / omega_m_f
steps = int(round(T_mod / dt))

if np.abs(nu_m_f - nu_m_i) < 0.01:
    N_turn = int(round((0.025 * N) / (2 * np.pi * f1 * b_target * steps)))
else:
    N_turn = int(round(np.abs(nu_m_f - nu_m_i) * N * omega_s) / (2 * np.pi * f2 * b_target * steps))

print(N_turn)

n_steps = steps * N_turn

t = 0.0

# ----------- lambda functions -----------

percent = 0.1

T_tot = n_steps * dt
T_percent = percent * T_tot

omega_lambda = lambda t: (
    omega_m_i if t <= T_percent
    else omega_m_i + (Delta_omega) * ((t - T_percent) / (T_tot - T_percent))
    if t < T_tot
    else omega_m_f
)

epsilon_function = lambda t: (
    epsilon_i + Delta_eps * (t - T_percent) / (T_tot - T_percent)
    if t < T_tot
    else epsilon_f)
 
epsilon = lambda t: (
    epsilon_i * (t / T_percent) if t <= T_percent
    else epsilon_function(t)
)

a_lambda = lambda t: epsilon(t) / (omega_lambda(t)/omega_s)

#a_lambda = lambda t: 0
"""
omega_lambda = lambda t: omega_m_f
epsilon_function = lambda t: (
    epsilon_i + Delta_eps * (t / T_tot)
    if t < T_tot
    else epsilon_f)

a_lambda = lambda t: epsilon_function(t) / (omega_lambda(t)/omega_s)"""


