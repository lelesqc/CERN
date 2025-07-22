import numpy as np
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon, MultiPolygon

xy = np.load(f"../code/tune_stuff/island_particles.npz")

x = xy['x']
y = xy['y']

XY = np.stack((x, y), axis=-1)  # shape: (n_pts, n_particles, 2)
XY_to_plot = np.array(XY)

concave_polygon = XY_to_plot

x_flat = x.flatten()
y_flat = y.flatten()

points = np.column_stack((x_flat, y_flat))

alpha = 1.0 
polygon = alphashape.alphashape(points, alpha)

if not polygon.is_valid:
    polygon = polygon.buffer(0)

area = polygon.area
print("Area:", area)

plt.scatter(points[:, 0], points[:, 1], s=1, color='blue')

if isinstance(polygon, Polygon):
    plt.plot(*polygon.exterior.xy, color='red', label='Concave Hull')
    plt.fill(*polygon.exterior.xy, alpha=0.3, color='green')
elif isinstance(polygon, MultiPolygon):
    for poly in polygon.geoms:
        plt.plot(*poly.exterior.xy, color='red')
        plt.fill(*poly.exterior.xy, alpha=0.3, color='green')

plt.title(f"Area: {area:.2f}")
plt.show()
