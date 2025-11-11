import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipk, ellipe
from sage.functions.jacobi import inverse_jacobi, jacobi

import params_fcc as par

# ------------------ functions -------------------------

def H0_for_action_angle(q, p):
    Q = (q + np.pi) / par.lambd
    P = par.lambd * p
    return 0.5 * P**2 - par.A**2 * np.cos(par.lambd * Q)

def compute_action_angle(kappa_squared, P):
    action = 8 * par.A / np.pi * (ellipe(kappa_squared) - (1 - kappa_squared) * ellipk(kappa_squared))
    K_of_kappa = (ellipe(kappa_squared) - (np.pi * action)/(8 * par.A)) / (1 - kappa_squared)
    Omega = np.pi / 2 * (par.A / K_of_kappa)
    x = P / (2 * np.sqrt(kappa_squared) * par.A)
    
    u = inverse_jacobi('cn', float(x), float(kappa_squared))
    theta = (Omega / par.A) * u
    return action, theta

def Delta_q(p, psi, t, dt):
    return par.lambd**2 * p * dt + par.a_lambda(par.t) * par.omega_lambda(t) * np.cos(psi) * dt

def hamiltonian(q, p):
    H0 = 0.5 * par.lambd**2 * p**2 + par.A**2 * np.cos(q)    
    H1 = par.a_lambda(par.t) * par.omega_lambda(par.t) * np.cos(par.omega_lambda(par.t) * par.t + par.phi_0) * p
    
    return H0 + H1
    
def dV_dq(q): 
    return par.A**2 * np.sin(q) 

def compute_action_angle_inverse(X, Y):
    action = (X**2 + Y**2) / (2)
    theta = np.arctan2(-Y, X)
    return action, theta

def compute_Q_P(theta, Omega, kappa_squared):
    Q = 2 / par.lambd * np.arccos(jacobi('dn', float(par.A * theta / Omega), float(kappa_squared))) * np.sign(np.sin(theta))
    P = 2 * np.sqrt(kappa_squared) * par.A * jacobi('cn', float(par.A * theta / Omega), float(kappa_squared))
    return Q, P

def compute_phi_delta(Q, P):
    delta = P / par.lambd
    phi = par.lambd * Q - np.pi
    return phi, delta

def integrator_step(q, p, psi, t, dt, Delta_q, dV_dq): 
    #print(par.a_lambda(par.t))
    #par.damp_rate = 0
    #par.D = 0

    q += Delta_q(p, psi, t, dt/2)
    q = np.mod(q, 2 * np.pi)        
    t_mid = t + dt/2
    p += dt * dV_dq(q) - dt * 2 * par.damp_rate * p / par.beta**2 + np.sqrt(dt) * par.D * np.random.normal(0, 1, size=p.shape) 
    q += Delta_q(p, psi, t_mid, dt/2)
    q = np.mod(q, 2 * np.pi)     

    return q, p

def find_h0_numerical(I_target):
    def G_objective(h0_val):
        m = 0.5 * (1 + h0_val / par.A**2)
        epsilon = 1e-12
        m = np.clip(m, epsilon, 1 - epsilon)
        return (8 * par.A / np.pi) * (ellipe(m) - (1 - m) * ellipk(m)) - I_target

    epsilon_h = 1e-9 * par.A**2
    return brentq(G_objective, -par.A**2 + epsilon_h, par.A**2 - epsilon_h)

def birkhoff_average(phase_advances, n=1):
    N = len(phase_advances)
    k = np.arange(1, N)
    t = k/N
    weights = np.exp(-1/(t**n * (1-t)**n))
    weights[t <= 0] = 0
    weights[t >= 1] = 0
    norm = np.sum(weights)
    return np.sum(weights * phase_advances[1:]) / norm

def avg_phase_adv_runtime(prev_angle_unwrapped, current_angle, prev_tune, count):
    # Calcolo incremento angolare con unwrapping
    delta_raw = current_angle - (prev_angle_unwrapped % (2*np.pi))
    delta_angle = np.angle(np.exp(1j*delta_raw))  # differenza "continua" ∈ (-π, π)
    current_angle_unwrapped = prev_angle_unwrapped + delta_angle

    # Conversione in unità di tune
    delta_tune = delta_angle / (2*np.pi) * par.omega_s

    # Aggiorna media ricorsiva
    new_tune = prev_tune + (delta_tune - prev_tune) / (count + 1)
    new_count = count + 1

    return new_tune, new_count, current_angle_unwrapped

def H_resonant(q, p, x, y, x0, y0, k2):
    psi = par.omega_lambda(par.t) * par.t
    P = par.lambd * p
    action, angle = compute_action_angle(k2, P)
    k_prime = np.sqrt(1 - k2)
    q_param = np.exp(- np.pi * ellipk(k_prime**2) / ellipk(k2))
    G_of_I = 2 * np.pi * par.A / (ellipk(k2) * par.lambd) * np.sqrt(q_param) / (1 + q_param) 
    I = ((x - x0)**2 + (y - y0)**2) / 2

    psi = np.mod(par.omega_lambda(par.t) * par.t, 2*np.pi)

    return H0_for_action_angle(q, p) + par.epsilon_function(par.t) * G_of_I * np.cos(angle - psi) - I * par.omega_lambda(par.t)