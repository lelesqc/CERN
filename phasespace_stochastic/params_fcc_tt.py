import numpy as np
import yaml
import os

class Params:
    def __init__(self, config_path="params.yaml"):
        modulation = os.environ.get("MODULATION")
        thermal_bath = os.environ.get("THERMAL_BATH")      

        # ------------ machine ----------------
        self.bending_radius = 14430 
        self.h = 130000
        self.C_gamma = 8.85e-5    # m * GeV^-3
        
        # --- AGGIORNATI PER FCC-ee ttbar (Baseline FS 2505.00274) ---
        self.nu_s = 0.066                         # Synchrotron tune per ttbar
        self.T_rev = 302.54e-6
        self.V = 11.7e9                           # RF Voltage mostruoso (11.7 GV)
        self.radius = 10.76e3
        self.mc2 = 0.511e6
        self.gamma = 357142.8                     # Gamma per E = 182.5 GeV
        self.momentum_compaction = 2.8e-6         # alpha_c ridotta al minimo
        self.U_0 = 10.4e9                         # Energy loss (10.4 GeV a giro!)
        # -----------------------------------------
        
        self.damping_part_number = self.momentum_compaction * self.bending_radius / self.radius
        self.E_s = self.gamma * self.mc2
        self.eta = self.momentum_compaction - 1/self.gamma**2
        self.omega_rev = 2 * np.pi / self.T_rev

        # -------------- model -----------------

        if thermal_bath == "no":
            self.damp_rate = 0
        else:
            self.damp_rate = self.U_0 / (2 * self.T_rev * self.E_s) * (2 + self.damping_part_number)

        self.beta = np.sqrt(1 - 1/self.gamma**2)
        
        self.N = 100
        self.N_turn = 250
        
        self.phi_0 = 0.0
        self.e = 1
        self.lambd = np.sqrt(self.h * self.eta * self.omega_rev)
        self.omega_s = self.omega_rev * np.sqrt(self.e * self.h * self.V * self.eta / (2 * np.pi * self.E_s * self.beta**2))
        self.A = self.omega_s / self.lambd
        self.Cq = 3.83e-13
        #self.Cq = 0
        self.D = self.gamma / self.beta**3 * np.sqrt(self.damp_rate * self.Cq / self.radius)
        self.temperature = self.gamma**2 * self.h * self.eta * self.omega_rev * self.Cq / (2 * (2 + self.damping_part_number) * self.beta**4 * self.radius) 

        # -------------- YAML ------------------

        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.epsilon = config["epsilon"]
        self.nu_m = config["nu_m"]
        self.sigma = config.get("sigma", 1.0)

        # ------------- variables -----------------
        
        if modulation == "no":
            self.a = 0
        else:
            self.a = self.epsilon / self.nu_m

        self.omega_m = self.nu_m * self.omega_s
        self.T_s = 2 * np.pi / self.omega_s
        self.dt = self.T_s / self.N
        self.T_mod = 2 * np.pi / self.omega_m
        self.steps = int(round(self.T_mod / self.dt))
        self.n_steps = self.steps * self.N_turn
        self.t = 0.0