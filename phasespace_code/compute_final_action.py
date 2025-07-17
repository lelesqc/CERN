import numpy as np
import params as par
import matplotlib.pyplot as plt

grid_lim = 12.9
n_particles = 80

xy_data = np.load(f"action_angle/phasespace_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
tunes_data = np.load(f"tune_analysis/tunes_results.npz")

x_init = np.linspace(-0.1, grid_lim, n_particles)

x = xy_data['x']
y = xy_data['y']
tunes_list = tunes_data['tunes_list']

mask = (tunes_list > 0.78) & (tunes_list < 0.81)
x = x[:, mask]
y = y[:, mask]
x_init = x_init[mask]

center = (9.93, -0.27)
mask_x = (x_init >= center[0]) & (x_init <= np.max(x_init))

x_init = x_init[mask_x]

print(x_init.shape)

x = x[:, mask_x]
y = y[:, mask_x]

print(x.shape, y.shape)

plt.scatter(x, y, s=3, label="Phase Space for final distr.", alpha=1.0)
plt.show()