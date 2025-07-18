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
    #print(f"{t:.3f}, {par.a_lambda(t):.5f}, {np.cos(psi):2f}")
    return par.lambd**2 * p * dt + par.a_lambda(t) * par.omega_lambda(t) * np.cos(psi) * dt

def Delta_q_fixed(p, psi, a, omega_m, dt):
    #print(np.cos(psi))

    return par.lambd**2 * p * dt + a * omega_m * np.cos(psi) * dt

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
    q += Delta_q(p, psi, t, dt/2)
    q = np.mod(q, 2 * np.pi)        
    t_mid = t + dt/2
    p += dt * dV_dq(q)
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
    
def compute_angle(q, p):    
    return np.arctan2(p, q)

def pts_in_section(points, x_c, y_c, x_A, y_A, R=None):
    points = np.array(points)  # Converti in numpy array se non lo è già
    x, y = points[:, 0], points[:, 1]

    # Calcola angoli dei punti di bordo
    dx_A = x_A - x_c
    theta_A = np.arctan2(y_A, dx_A)
    theta_min = -theta_A
    theta_max = theta_A

    # Calcola angoli e distanze dei punti
    dx = x - x_c
    dy = y - y_c
    theta = np.arctan2(dy, dx)
    r = np.sqrt(dx**2 + dy**2)

    # Filtra i punti
    angle_mask = (theta >= theta_min) & (theta <= theta_max)
    if R is not None:
        radius_mask = r <= R
        mask = angle_mask & radius_mask
    else:
        mask = angle_mask

    return points[mask]