import numpy as np 
import matplotlib.pyplot as plt
import params_fcc as par

data = np.load("ipac_simulations/a_0.03/nu_0.94/final_distr_cen.npz")

x = data["x"]
y = data["y"]

"""plt.scatter(x, y, s=1)
plt.show()"""

print(par.omega_s, par.omega_m)