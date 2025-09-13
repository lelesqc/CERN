import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipk, ellipe
from sage.functions.jacobi import inverse_jacobi, jacobi

import params as par

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
    
def dV_dq(q): 
    return par.A**2 * np.sin(q) 

def Delta_q(p, psi, t, dt):
    #print(f"{t:.3f}, {np.cos(psi)}, {par.a_lambda(t):.5f}, {par.omega_lambda(t)/par.omega_s:.5f}")
    val_q0 = par.damp_rate
    val_dq = dt * 2 * par.damp_rate * p[0] / par.beta**2

    if np.isnan(val_q0) or np.isnan(val_dq):
        print("NaN detected! q[0]:", val_q0, "Delta_q[0]:", val_dq, )
        import sys; sys.exit(1)  # Ferma l'esecuzione

    #print(val_q0, val_dq)
    return par.lambd**2 * p * dt + par.a * par.omega_m * np.cos(psi) * dt

def hamiltonian(q, p):
    H0 = 0.5 * par.lambd**2 * p**2 + par.A**2 * np.cos(q)    
    H1 = par.a * par.omega_m * np.cos(par.omega_m * par.t + par.phi_0) * p
    
    return H0 + H1

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
    noise_D = par.gamma / par.beta**2 * np.sqrt(2 * par.damp_rate * par.h * par.eta * par.Cq / par.radius)
    
    q += Delta_q(p, psi, t, dt/2)
    q = np.mod(q, 2 * np.pi)        
    t_mid = t + dt/2
    p += dt * dV_dq(q) - dt * 2 * par.damp_rate * p / par.beta**2 + np.sqrt(dt) * noise_D * np.random.normal(0, 1, size=p.shape) 
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
