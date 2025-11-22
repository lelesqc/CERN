"""import numpy as np
import matplotlib.pyplot as plt
import params as par
from scipy.special import ellipk

data = np.load("actions_stuff/actions_first_part.npz")
actions = data['init_actions']

kappa2 = 1
g = 1
q = np.exp(- np.pi * ellipk(1 - kappa2) / ellipk(kappa2))
delta = 1 / par.T_tot
omega_prime = 0.1
A = 2 * np.pi * par.A / (par.lambd * ellipk(kappa2)) * np.sqrt(q) / (1 + q)
#factor_prime = 

pr = np.sqrt(par.epsilon) * 8 / np.pi * g / omega_prime"""

import numpy as np
from scipy.special import ellipk, ellipe
from scipy.optimize import bisect
import params as par

def neishtadt_probability_local(omega0, omega_i, omega_f, omega_s, domega_dtau, epsilon):
    # trova tau* (frequenza di modulazione = omega_s)
    frac = (omega_s - omega_i) / (omega_f - omega_i)
    if frac < 0 or frac > 1:
        raise ValueError("La risonanza non viene attraversata nel range selezionato.")
    
    omega_star = omega_i + frac * (omega_f - omega_i)

    # trova κ* tale che Ω(κ*) = ω_star
    def Omega(k):
        return np.pi * omega0 / (2 * ellipk(k**2))
    k_star = bisect(lambda k: Omega(k) - omega_star, 1e-6, 0.9999)

    # calcola A e g al punto risonante
    def A_g(k):
        K = ellipk(k**2)
        E = ellipe(k**2)
        kp = np.sqrt(1 - k**2)
        Kp = ellipk(kp**2)
        q = np.exp(-np.pi * Kp / K)
        A = 4 * np.sqrt(q) / (1 + q)
        g = - (np.pi**2) / (16.0 * k**2 * (1 - k**2)) * (E - (1 - k**2) * K) / (K**3)
        return A, g

    A_star, g_star = A_g(k_star)

    # derivata numerica locale con piccolo incremento in frequenza
    delta = 1e-4 * omega_star
    k_plus = bisect(lambda k: Omega(k) - (omega_star + delta), 1e-6, 0.9999)
    k_minus = bisect(lambda k: Omega(k) - (omega_star - delta), 1e-6, 0.9999)
    A_plus, g_plus = A_g(k_plus)
    A_minus, g_minus = A_g(k_minus)

    d_sqrtAg_dtau = ((np.sqrt(-A_plus/g_plus) - np.sqrt(-A_minus/g_minus)) / (2*delta)) * domega_dtau

    Pr = np.sqrt(epsilon) * (8 / np.pi) * g_plus / domega_dtau * d_sqrtAg_dtau


# === ESEMPIO DI UTILIZZO ===
epsilon = 1 / (par.omega_s * (par.T_tot - par.T_percent))
omega0 = par.A             # frequenza sincrona (ω_s)
omega_i = 0.9 * omega0   # frequenza di modulazione iniziale
omega_f = 0.8 * omega0   # frequenza di modulazione finale
domega_dtau = 1 / epsilon * ((omega_f - omega_i) * (1 / (par.T_tot - par.T_percent)))      # derivata lenta della frequenza (negativa -> frequenza in diminuzione)

Pr = neishtadt_probability_local(par.A, omega_i, omega_f, omega0, domega_dtau, epsilon)
print(f"Probabilità teorica di intrappolamento P_r = {Pr:.4f}")
