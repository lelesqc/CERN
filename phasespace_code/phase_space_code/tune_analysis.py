import numpy as np
import matplotlib.pyplot as plt

import params as par

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
    poly = polygon.copy()
    triangles = []
    
    while len(poly) > 3:
        ear_found = False
        for i in range(len(poly)):
            if is_ear(poly, i):
                a, b, c = poly[(i-1)%len(poly)], poly[i], poly[(i+1)%len(poly)]
                triangles.append([a, b, c])
                del poly[i]
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

def calculate_polygon_area_simple(points):
    """Calcola l'area usando la formula del laccio (shoelace formula)"""
    points = np.array(points)
    
    # Ordina i punti in senso antiorario rispetto al centroide
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    # Formula del laccio
    x = sorted_points[:, 0]
    y = sorted_points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# -------------------------------------------------------------------

data = np.load("tune_analysis/fft_results.npz")
action_angle = np.load(f"action_angle/tune_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

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
    if tunes_list_pos[i] < 0.82:
        XY_to_plot.append((x_fin[i], y_fin[i]))
        tunes_to_plot.append(tunes_list_pos[i])

XY_to_plot = np.array(XY_to_plot)
tunes_to_plot = np.array(tunes_to_plot)

# ----------------------------------------------------

concave_polygon = XY_to_plot

# Calcola l'area
area = calculate_polygon_area_simple(concave_polygon)
print(f"Area del poligono: {area}")

# Visualizzazione
poly = np.array(concave_polygon + [concave_polygon[0]])
plt.plot(poly[:,0], poly[:,1], 'bo-')
plt.fill(poly[:,0], poly[:,1], alpha=0.3)
plt.title(f"Poligono concavo - Area: {area:.2f}")
plt.grid(True)
plt.show()
