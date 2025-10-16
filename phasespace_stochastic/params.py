import numpy as np
import yaml

class Params:
    def __init__(self, config_path="params.yaml"):
        # ------------ machine ----------------
        self.h = 328
        self.C_gamma = 8.85e-5    # m * GeV^-3
        self.nu_s = 0.0075
        self.T_rev = 0.66e-6
        self.V = 1.5e6
        self.radius = 4.01
        self.mc2 = 0.511e6
        self.gamma = 2935.42
        self.gamma_transition = 26.44
        self.E_s = self.gamma * self.mc2
        self.eta = 1/self.gamma_transition**2 - 1/self.gamma**2
        self.U_0 = self.C_gamma * (1e-9)**3 * self.E_s**4 / self.radius
        self.omega_rev = 2 * np.pi / self.T_rev
        self.k_B = 8.617333262e-5

        # -------------- model -----------------
        self.k_lele_als = 4.54e6
        self.damp_rate = self.U_0 / self.T_rev / self.E_s
        self.beta = np.sqrt(1 - 1/self.gamma**2)
        self.N = 100
        self.N_turn = 675    # 15 volte il damping time
        #self.N_turn = 2500
        self.phi_0 = 0.0
        self.e = 1
        self.lambd = np.sqrt(self.h * self.eta * self.omega_rev)
        self.omega_s = self.omega_rev * np.sqrt(self.e * self.h * self.V * self.eta / (2 * np.pi * self.E_s * self.beta**2))
        self.A = self.omega_s / self.lambd
        self.Cq = 3.83e-13

        # -------------- YAML ------------------
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.epsilon = config["epsilon"]
        self.nu_m = config["nu_m"]

        # ------------- variables -----------------
        self.omega_m = self.nu_m * self.omega_s
        self.a = self.epsilon / self.nu_m
        #self.a = 0
        self.T_s = 2 * np.pi / self.omega_s
        self.dt = self.T_s / self.N
        self.T_mod = 2 * np.pi / self.omega_m
        self.steps = int(round(self.T_mod / self.dt))
        self.n_steps = self.steps * self.N_turn
        self.t = 0.0

    def update_dependent(self):
        self.eta = 1/self.gamma_transition**2 - 1/self.gamma**2
        self.beta = np.sqrt(1 - 1/self.gamma**2)

# USO:
# import params
# par = params.Params()
# par.gamma = nuovo_valore
# par.update_dependent()

#print((1 / (2 * Params().damp_rate) * 10) / Params().dt)
#print(58000/125)