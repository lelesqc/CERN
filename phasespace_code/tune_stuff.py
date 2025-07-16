import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import params as par

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def cross(o, a, b):
    """Calcola il prodotto vettoriale 2D tra OA e OB"""
    return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])

def is_ear(poly, i):
    """Verifica se il vertice i è un 'ear' (orecchio)"""
    n = len(poly)
    a, b, c = poly[(i-1)%n], poly[i], poly[(i+1)%n]
    
    # Controlla che l'angolo sia convesso
    if cross(a, b, c) <= 0:
        return False
    
    # Controlla che nessun altro punto sia dentro il triangolo
    triangle = [a, b, c]
    for j in range(n):
        if j not in [(i-1)%n, i, (i+1)%n]:
            point = poly[j]
            if point_in_triangle(point, triangle):
                return False
    return True

def point_in_triangle(p, triangle):
    """Verifica se il punto p è dentro il triangolo"""
    a, b, c = triangle
    d1 = cross(p, a, b)
    d2 = cross(p, b, c)
    d3 = cross(p, c, a)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def triangulate_polygon(polygon):
    """Triangola un poligono concavo usando l'algoritmo ear clipping"""
    # Converti l'array numpy in una lista di tuple
    poly = [tuple(point) for point in polygon]
    triangles = []
    
    while len(poly) > 3:
        ear_found = False
        for i in range(len(poly)):
            if is_ear(poly, i):
                a, b, c = poly[(i-1)%len(poly)], poly[i], poly[(i+1)%len(poly)]
                triangles.append([a, b, c])
                del poly[i]  # Ora funziona perché poly è una lista
                ear_found = True
                break
        
        if not ear_found:
            print("Poligono non semplice o non valido per la triangolazione")
            break
    
    if len(poly) == 3:
        triangles.append(poly)
    
    return triangles

def calculate_polygon_area(polygon):
    """Calcola l'area di un poligono concavo usando la triangolazione"""
    triangles = triangulate_polygon(polygon)
    total_area = 0.0
    
    for triangle in triangles:
        a, b, c = triangle
        # Area del triangolo usando il prodotto vettoriale
        area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
        total_area += area
    
    return total_area

# -------------------------------------------------------------------

data = np.load("tune_analysis/fft_results.npz")
action_angle_evolution = np.load(f"../code/action_angle/none_a0.025-0.050_nu0.90-0.80.npz")
integrator = np.load(f"../code/integrator/evolved_qp_none.npz")

actions = action_angle_evolution['actions']

psi = integrator['psi']
x = action_angle_evolution['x']
y = action_angle_evolution['y']

x = x[::10, :]
y = y[::10, :]
actions = actions[::10, :]

xy_pairs = list(zip(x[:, 0], y[:, 0]))
xy_pairs = np.array(xy_pairs)

points_per_trajectory = 125
total_points = len(xy_pairs)
n_trajectories = total_points // points_per_trajectory

print(f"Punti totali: {total_points}")
print(f"Numero di traiettorie: {n_trajectories}")
print(f"Punti per traiettoria: {points_per_trajectory}")

# Separa le traiettorie
trajectories = []
for i in range(n_trajectories):
    start_idx = i * points_per_trajectory
    end_idx = (i + 1) * points_per_trajectory
    trajectory = xy_pairs[start_idx:end_idx]
    trajectories.append(trajectory)

trajectories = np.array(trajectories)

plt.scatter(trajectories[:, 0, 0], trajectories[:, 0, 1], s=1)
plt.show()

actions_computed = []

for _, trajectory in enumerate(trajectories):
    concave_area_triang_i = calculate_polygon_area(trajectory)
    actions_computed.append(concave_area_triang_i / (2 * np.pi))
    
#concave_area_triang = calculate_polygon_area(xy_pairs)
#print(f"Area of the concave hull: {concave_area_triang:.4f}")


"""
x = action_angle['x']
y = action_angle['y']
actions = action_angle['actions_list']

x_fin = x[-1, :] 
y_fin = y[-1, :]           
actions_init = actions[0, :] 

spectra = data['spectra']
freqs_list = data['freqs_list']
tunes_list = data['tunes_list']

actions_init_pos = actions_init
tunes_list_pos = tunes_list

XY_to_plot = []
tunes_to_plot = []

for i in range(len(actions_init_pos)):
    if tunes_list_pos[i] < 0.82 and x_fin[i]**2 + y_fin[i]**2 > 2.5:
        XY_to_plot.append((x_fin[i], y_fin[i]))
        tunes_to_plot.append(tunes_list_pos[i])

XY_to_plot = np.array(XY_to_plot)
tunes_to_plot = np.array(tunes_to_plot)

edges = alpha_shape(XY_to_plot, alpha=0.75, only_outer=True)

concave_area_triang = calculate_polygon_area(XY_to_plot)
print(f"Area of the concave hull: {concave_area_triang:.4f}")

plt.figure(figsize=(10, 8))

# Plotta i punti colorati per tune
plt.scatter(XY_to_plot[:, 0], XY_to_plot[:, 1], c=tunes_to_plot, cmap='viridis', s=30, alpha=0.7)
plt.colorbar(label='Tune')

# Plotta gli edge dell'alpha shape
for edge in edges:
    i, j = edge
    x_coords = [XY_to_plot[i, 0], XY_to_plot[j, 0]]
    y_coords = [XY_to_plot[i, 1], XY_to_plot[j, 1]]
    plt.plot(x_coords, y_coords, 'r-', linewidth=1, alpha=0.8)

print(x.shape)
"""