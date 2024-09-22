import openmesh
import numpy as np

def angle_between_vectors(v1, v2):
    dv1 = np.linalg.norm(v1)
    dv1 = max(dv1,1e-10)
    dv2 = np.linalg.norm(v2)
    dv2 = max(dv2,1e-10)

    return np.arccos(np.dot(v1,v2)/(dv1*dv2))


def compute_vertex_weighted_normals(mesh: openmesh.TriMesh):
    points = mesh.points()
    normal_faces = np.zeros((mesh.n_faces(), 3))
    face_angles = np.zeros(mesh.n_faces(), dtype=dict)

    for face in mesh.faces():
        v_it = openmesh.FaceVertexIter(mesh, face)

        v0 = next(v_it).idx()
        v1 = next(v_it).idx()
        v2 = next(v_it).idx()

        difv1v0 = points[v1] - points[v0]
        difv2v0 = points[v2] - points[v0]

        normal = np.cross(difv1v0, difv2v0)
        normal /= np.linalg.norm(normal)
        normal_faces[face.idx()] = normal
        face_angles[face.idx()] = {
            v0: angle_between_vectors(difv1v0, difv2v0),
            v1: angle_between_vectors(points[v2] - points[v1], -difv1v0),
            v2: angle_between_vectors(-difv2v0, points[v1] - points[v2])
        }

    normal_vertices = np.zeros((mesh.n_vertices(), 3))
    for vertex in mesh.vertices():
        face_it = openmesh.VertexFaceIter(mesh, vertex)
        for face in face_it:
            normal_vertices[vertex.idx(), :] += normal_faces[face.idx(), :]*face_angles[face.idx()][vertex.idx()]
        normal_vertices[vertex.idx(), :] /= np.linalg.norm(normal_vertices[vertex.idx(), :])
    
    return normal_vertices
