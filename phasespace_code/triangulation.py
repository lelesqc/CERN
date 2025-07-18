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


"""
def alpha_shape(points, alpha):
    tri = Delaunay(points)
    edges = set()
    # Itera su tutte le triplette di punti (simplessi)
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]
        # Calcola la circonferenza circoscritta
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area) if area != 0 else np.inf
        # Filtra i simplessi con raggio minore di alpha
        if circum_r < alpha:
            edges.add((ia, ib))
            edges.add((ib, ic))
            edges.add((ia, ic))
    return edges

def concave_hull_area(points, alpha=0.5):
    points = np.array(points)
    edges = alpha_shape(points, alpha)
    
    # Costruisci il grafo dei bordi
    from collections import defaultdict
    graph = defaultdict(list)
    for ia, ib in edges:
        graph[ia].append(ib)
        graph[ib].append(ia)
    
    # Trova il percorso del bordo (algoritmo semplice, potrebbe non essere perfetto)
    boundary = []
    if len(graph) > 0:
        start = min(graph.keys())
        boundary.append(start)
        current = start
        while True:
            next_node = None
            for neighbor in graph[current]:
                if neighbor not in boundary:
                    next_node = neighbor
                    break
            if next_node is None:
                break
            boundary.append(next_node)
            current = next_node
    
    # Estrai i punti del bordo in ordine
    hull_points = points[boundary]
    
    # Crea un poligono con Shapely e calcola l'area
    polygon = Polygon(hull_points)
    return polygon.area"""