import numpy as np
import polyscope as ps
import openmesh
import argparse
from edge_normals import compute_vertex_curvature, compute_vertex_weighted_curvature

# ps.set_up_dir("z_up")
# ps.set_front_dir("x_front")

# ps.init()

# # generate some random points
# points = np.random.rand(1000, 3)
# vals = np.random.rand(1000)
# vecs = np.random.rand(1000, 3)

# # visualize!
# ps_cloud = ps.register_point_cloud("noise", points)
# ps_cloud.add_scalar_quantity("vals", vals)
# ps_cloud.add_vector_quantity("vecs", vecs)

# ps.show()

###########################################

# import openmesh
# import argparse
# import polyscope as ps
# # from random import sample
# import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument("--file", type=str, default="", help="Nombre del archivo")
# opt = parser.parse_args()

# mesh = openmesh.read_trimesh(opt.file)

# mask = np.random.choice([False, True], len(mesh.points()), p=[0.8, 0.2])
# point_cloud = np.array([mesh.points()[i] for i in range(len(mask)) if mask[i]])

# ps.init()
# ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
# ps_cloud = ps.register_point_cloud("our points", point_cloud)
# ps.show()

###########################################


# import numpy as np
# import polyscope as ps

# ps.set_up_dir("z_up")
# n = 10
# def compute_normal_from_angle(phi, theta):
#     return np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

# class Circle:
#     def __init__(self, center, radius, normal):
#         self.c = np.array(center)
#         self.r = radius
#         self.n = np.array(normal)
#         self.n = self.n/np.linalg.norm(self.n)  # normalized here so we don't have to worry

#     def compute_node_and_edges(self, n_nodes=10):
#         # let v1, v2, n an orthonormal system

#         # get v1 so <v1, n> == 0
#         v1 = np.array([self.n[1], -self.n[0], 0])
#         if np.array_equal(v1, np.zeros(3)):
#             v1 = np.array([1, 0, 0])

#         # v2 = n x v1
#         v2 = np.cross(self.n, v1)

#         # make them orthonormal
#         v1 = v1/np.linalg.norm(v1)
#         v2 = v2/np.linalg.norm(v2)

#         # sample
#         nodes = []
#         edges = []
#         for i in range(0, n_nodes):
#             theta = 2 * np.pi * (i / n_nodes)
#             nodes.append(self.c + self.r * (v1 * np.cos(theta) + v2 * np.sin(theta)))
#             edges.append([i, (i+1) % n_nodes])

#         return np.array(nodes), np.array(edges)

# our_circle = Circle([0, 0, 3], 10, compute_normal_from_angle(3*np.pi/4, np.pi/8))
# circle_nodes, circle_edges = our_circle.compute_node_and_edges(n)


# ps.init()
# ps_axis = ps.register_point_cloud("Axis", np.array([[0, 0, 0]]))
# ps_axis.add_vector_quantity("x", np.array([[1, 0, 0]]))
# ps_axis.add_vector_quantity("y", np.array([[0, 1, 0]]))
# ps_axis.add_vector_quantity("z", np.array([[0, 0, 1]]))


# ps_cloud = ps.register_point_cloud("Supporting Circle Centers", np.array([our_circle.c]))
# ps_circle = ps.register_curve_network("Circle", circle_nodes, circle_edges, radius=0.001)
# # Además, podemos mostrarl la normal del círculo
# ps_cloud.add_vector_quantity("Normal", np.array([our_circle.n]))

# edge_normals_pos = []
# edge_normals = []
# for i in range(n):
#     # Punto medio del edge
#     med_p = (circle_nodes[i] + circle_nodes[(i+1) % n]) * 0.5
#     edge_normals_pos.append(med_p)
    
#     # Vector del edge
#     edge_vec = circle_nodes[(i+1) % n] - circle_nodes[i]
    
#     # Normal del edge usando el producto cruzado con la normal del círculo
#     edge_normal = np.cross(edge_vec, our_circle.n)
    
#     # Normalizar la normal
#     edge_normal = edge_normal / np.linalg.norm(edge_normal)
    
#     edge_normals.append(edge_normal)

# ps_cloud_2 = ps.register_point_cloud("Edge Normals Pos", np.array(edge_normals_pos))
# ps_cloud_2.add_vector_quantity("Edge Normals", np.array(edge_normals))


# ps.show()

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="", help="Nombre del archivo")
opt = parser.parse_args()

mesh = openmesh.read_trimesh(opt.file)
vertex_curvatures = compute_vertex_weighted_curvature(mesh)
print(min(vertex_curvatures))
print(max(vertex_curvatures))
ps.init()
ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
ps_mesh.add_scalar_quantity("Vertex curvatures", vertex_curvatures)
ps.show()