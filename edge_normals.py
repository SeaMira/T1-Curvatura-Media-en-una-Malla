import openmesh
import numpy as np
from vertex_normals import compute_vertex_weighted_normals, angle_between_vectors

def compute_edge_normals(mesh: openmesh.TriMesh):
    points = mesh.points()
    vertex_normals = compute_vertex_weighted_normals(mesh)
    vertex_to_edge_normals = [{} for _ in range(mesh.n_vertices())]    
    for vertex in mesh.vertices():
        for neighbor in mesh.vv(vertex):
            v = points[vertex.idx()]
            Nv = vertex_normals[vertex.idx()]
            t = points[neighbor.idx()]
            Nt = vertex_normals[neighbor.idx()]
            dif_t_v = t-v
            n_dif_t_v = np.linalg.norm(dif_t_v, ord=2)
            n_dif_t_v = max(n_dif_t_v, 1e-10)
            vertex_to_edge_normals[vertex.idx()][neighbor.idx()] = np.dot((Nt-Nv), dif_t_v)/(n_dif_t_v)

    return vertex_to_edge_normals


def compute_vertex_curvature(mesh: openmesh.TriMesh):
    vertex_curvatures = np.zeros(mesh.n_vertices())
    vertex_to_edge_normals = compute_edge_normals(mesh)
    for vertex in mesh.vertices():
        i = 0
        vertex_edges = vertex_to_edge_normals[vertex.idx()]
        for neighbor in mesh.vv(vertex):
            vertex_curvatures[vertex.idx()] += vertex_edges[neighbor.idx()]
            i += 1
        vertex_curvatures[vertex.idx()] /= max(i, 1)
    return vertex_curvatures



def compute_vertex_weighted_curvature(mesh: openmesh.TriMesh):
    vertex_curvatures = np.zeros(mesh.n_vertices())
    vertex_to_edge_normals = compute_edge_normals(mesh)
    points = mesh.points()

    for vertex in mesh.vertices():
        alpha = 0
        vertex_edges = vertex_to_edge_normals[vertex.idx()]

        neighbors = [neighbor for neighbor in mesh.vv(vertex)]
        n_neighbors = len(neighbors)

        if n_neighbors == 0:
            continue  

        for j in range(n_neighbors):
            neighbor = neighbors[j]
            next_neighbor = neighbors[(j+1) % n_neighbors]

            v = points[vertex.idx()]
            t = points[neighbor.idx()]
            r = points[next_neighbor.idx()]

            # Vectores de las aristas
            edge_vt = t - v
            edge_vr = r - v

            # Producto cruzado de las aristas
            cross_product = np.cross(edge_vt, edge_vr)

            # Normas de los vectores
            norm_vt = np.linalg.norm(edge_vt)
            norm_vt = max(norm_vt, 1e-10)
            norm_vr = np.linalg.norm(edge_vr)
            norm_vr = max(norm_vr, 1e-10)
            norm_cross = np.linalg.norm(cross_product)

            # Calcular el ángulo entre las aristas usando la fórmula proporcionada
            angle_vtr = np.arcsin(min(1.0, max(-1.0, norm_cross / (norm_vt * norm_vr))))

            vertex_curvatures[vertex.idx()] += angle_vtr * (vertex_edges[neighbor.idx()] + vertex_edges[next_neighbor.idx()])
            alpha += angle_vtr
            
        vertex_curvatures[vertex.idx()] /= max(alpha, 1e-10)
    return vertex_curvatures