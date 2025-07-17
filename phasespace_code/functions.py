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
    #print(f"{t:.3f}, {np.cos(psi)}, {par.a:.5f}, {par.omega_m/par.omega_s:.5f}")
    return par.lambd**2 * p * dt + par.a * par.omega_m * np.cos(psi) * dt

def compute_I_from_h0(h0, A):
    kappa_squared = 0.5 * (1 + h0 / (A**2))
    if kappa_squared < 0 or kappa_squared > 1:
        return np.inf
    K = ellipk(kappa_squared)
    E = ellipe(kappa_squared)
    I = (8 * A / np.pi) * (E - (1 - kappa_squared) * K)
    return I

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

def compute_angle(x, y, x_center=0.0, y_center=0.0):
    return np.arctan2(y - y_center, x - x_center)

def save_trajectory(q0, p0, q_center=0.0, p_center=0.0):
    q = q0
    p = p0

    angle_accumulated = 0.0
    trajectory = np.array([(q, p)])

def w_n(t, n):
    """Filtro analitico w^n(t)"""
    mask = (t > 0) & (t < 1)
    result = np.zeros_like(t)
    result[mask] = np.exp(-1 / (t[mask]**n * (1 - t[mask])**n))
    return result

def birkhoff_average(phase_advances, n=1):
    N = len(phase_advances)
    k = np.arange(1, N)
    t = k/N
    weights = np.exp(-1/(t**n * (1-t)**n))
    weights[t <= 0] = 0
    weights[t >= 1] = 0
    norm = np.sum(weights)
    return np.sum(weights * phase_advances[1:]) / norm

def shoelace(x_y):
    x_y = np.array(x_y)
    x_y = x_y.reshape(-1,2)

    x = x_y[:,0]
    y = x_y[:,1]

    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))

    area = .5*np.absolute(S1 - S2)

    return area

def order_points(points):
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
    order = np.argsort(angles)
    return points[order]


