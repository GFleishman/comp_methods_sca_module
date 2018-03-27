#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script computes the surface area, volume, and Willmore energy a triangular
mesh surface. The Willmore energy requires the principal curvatures at each
vertex. The principal curvatures are the eigenvectors/eigenvalues of the
second fundamental form matrix: an nxn (where n is the intrinsic dimension
of the surface) matrix that captures all information regarding local curvature
properties. I use the algorithm in:
Szymon Rusinkiewicz, "Estimating Curvatures and Their Derivatives on Triangle
Meshes" which (as of 03/27/2018) can be found here:
http://gfx.cs.princeton.edu/pubs/_2004_ECA/curvpaper.pdf
"""


import argparse
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
import pymesh
import numpy as np
import math


def parse_inputs():
    """Reads in the command line arguments using argparse module"""
    desc = """
    Compute the discrete Willmore energy, volume, and surface area
    of a mesh file. Write the mesh file out with Gaussian curvature
    scalar field at every vertex.
    """
    mandatory_arglist = ['mesh_input_path',
                         'mesh_output_path']
    mandatory_helplist = ['path to input mesh file',
                          'where to write the output mesh']
    parser = argparse.ArgumentParser(description=desc)
    for x in list(zip(mandatory_arglist, mandatory_helplist)):
        parser.add_argument(x[0], help=x[1])
    return parser.parse_args()


def rotation_matrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def load_mesh(mesh_path):
    """Loads mesh file, returns pymesh mesh object, vertices, and faces"""
    neuropil = pymesh.load_mesh(mesh_path)
    verts = neuropil.vertices
    faces = neuropil.faces
    return neuropil, verts, faces


def compute_edge_lengths(verts, faces):
    """computes length of every edge (line connecting two vertices)"""
    edges = np.empty((len(faces),) + (3, 3))
    for i, face in enumerate(faces):
        a = [verts[face[(ii+1)%3]]-verts[face[ii]] for ii in range(3)]
        edges[i] = np.array(a)
    return edges


def compute_face_properties(verts, faces, edges):
    """computes area of each face (triangle) and orthonormal coordinate system
    for each face, a normal vector and two orthogonal in plane vectors
    face_e1 and face_e2"""
    faces_e1 = edges[:, 0] + 0.5*edges[:, 1]
    face_normals = np.cross(faces_e1, edges[:, 1])
    face_areas = np.linalg.norm(face_normals, axis=1)/2
    face_normals /= face_areas[..., np.newaxis]*2
    faces_e1 /= np.linalg.norm(faces_e1, axis=1)[..., np.newaxis]
    faces_e2 = np.cross(face_normals, faces_e1)
    return face_normals, face_areas, faces_e1, faces_e2


def compute_vertex_properties(verts, faces, face_normals, face_areas):
    """computes orthonormal frame at each vertex:
    a normal vector (weighted sum of normals from 1-ring of faces touching
    vertex) and orthogonal vectors in plane orthognal to vertex normal"""
    vert_normals = np.zeros_like(verts).astype(np.float32)
    for i, face in enumerate(faces):
        weighted_normal = face_normals[i]*face_areas[i]
        for j, vert in enumerate(face):
            # this is doubley inefficient, computing each edge length twice
            vert_wn = weighted_normal/np.sum((verts[face[(j+1)%3]]-verts[vert])**2)
            vert_wn = weighted_normal/np.sum((verts[face[(j+2)%3]]-verts[vert])**2)
            vert_normals[vert] += vert_wn
    vert_normals /= np.linalg.norm(vert_normals, axis=1)[..., np.newaxis]

    verts_e1 = np.ones_like(verts).astype(np.float32)
    verts_e1[:, 1] = 0  # let x and y components of e1 be (1, 0)
    # uses nx * e1x + ny * e1y + nz * e1z = 0
    # just ensures verts_e1 are orthogonal to vertex normals
    verts_e1[:, 2] = -vert_normals[:, 0]/vert_normals[:, 2]
    verts_e1 /= np.linalg.norm(verts_e1, axis=1)[..., np.newaxis]
    verts_e2 = np.cross(vert_normals, verts_e1)
    return vert_normals, verts_e1, verts_e2


def estimate_2FF_at_faces(faces, edges, faces_e1, faces_e2, vert_normals):
    """estimates second fundamental form at each face using finite differences
    on the normal vector derivative in each of the three edge directions.
    components of 2FF are fit with least squares."""
    edge_diffs = np.empty((len(faces), 6))
    norm_diffs = np.empty((len(faces), 6))
    for i in range(3):
        edge_diffs[:, 2*i] = np.einsum('...i,...i', edges[:, i], faces_e1)
        edge_diffs[:, 2*i+1] = np.einsum('...i,...i', edges[:, i], faces_e2)
        diff = vert_normals[faces[:, (i+1)%3]] - vert_normals[faces[:, i]]
        norm_diffs[:, 2*i] = np.einsum('...i,...i', diff, faces_e1)
        norm_diffs[:, 2*i+1] = np.einsum('...i,...i', diff, faces_e2)

    dm = np.zeros((6, 3))
    def populate_dummy_matrix(dm, x):
        for i in range(3):
            dm[2*i, :2] = x[2*i:2*i+2]
            dm[2*i+1, 1:3] = x[2*i:2*i+2]
        return dm

    # least squares estimate for 2FF from vertex and norms finite diffs
    faces_2FF = np.empty_like(faces).astype(np.float64)
    for i, face in enumerate(faces):
        dm = populate_dummy_matrix(dm, edge_diffs[i])
        faces_2FF[i] = np.linalg.lstsq(dm, norm_diffs[i], rcond=None)[0]
    return faces_2FF


def estimate_2FF_at_vertices(verts, faces, face_areas, faces_2FF,
                             face_normals, faces_e1, faces_e2,
                             vert_normals, verts_e1, verts_e2):
    """computes weighted average of second fundamental forms for each face
    in 1-ring around vertex. 2FFs must be transformed to orthonormal basis
    for vertex (which includes a rotation to align normal vectors, and
    projection on the in plane basis vectors"""
    verts_2FF_weights = np.zeros_like(faces).astype(np.float64)
    for i, face in enumerate(faces):
        midpoints = [.5*(verts[face[ii]]+verts[face[(ii+1)%3]]) for ii in range(3)]

    # Theoretically, this commented out section should work to find voronoi
    # center of each triangle, but after several hours of careful inspection,
    # I simply cannot get the voronoi centers to look reasonable in paraview when
    # computed this way; the only explanation I have is that this must be a
    # numerially unstable method. I'm using the arithmetic mean of each triangle
    # instead.
    #    side_normals = [np.cross(face_normals[i], verts[face[(ii+1)%3]]-verts[face[ii]])
    #                    for ii in range(3)]
    #    dummy_matrix = np.concatenate((side_normals[0][..., np.newaxis],
    #                                  -side_normals[1][..., np.newaxis]), axis=1)
    #    b = midpoints[1] - midpoints[0]
    #    slopes = np.linalg.lstsq(dummy_matrix, b)[0]
    #    voronoi_center = 0.5*(midpoints[0] + slopes[0]*side_normals[0] +
    #                          midpoints[1] + slopes[1]*side_normals[1])

        voronoi_center = (1./3)*(verts[face[0]]+verts[face[1]]+verts[face[2]])
        for j in range(3):
            cp = np.cross(voronoi_center - verts[face[j]],
                          midpoints[j] - midpoints[(j-1)%3])
            verts_2FF_weights[i, j] = 0.5 * np.linalg.norm(cp) / face_areas[i]

    verts_2FF = np.zeros((verts.shape[0],) + (2, 2)).astype(np.float64)
    verts_weight_sums = np.zeros(len(verts))
    for i, face in enumerate(faces):
        for j, vert in enumerate(face):
            # correct for normal vector alignment
            theta = np.arccos(np.dot(face_normals[i], vert_normals[vert]))
            if np.allclose(theta, 0, rtol=1e-02, atol=1e-05):
                rot_matrix = np.eye(3)
            else:
                rot_axis = np.cross(vert_normals[vert], face_normals[i])
                rot_matrix = rotation_matrix(rot_axis, theta)

            # rotate and project vertex tangent vectors to face tangent vectors
            aa = np.einsum('...ij,...j->...i', rot_matrix, verts_e1[vert])
            bb1 = np.array([np.dot(aa, faces_e1[i]),
                            np.dot(aa, faces_e2[i])])
            aa = np.einsum('...ij,...j->...i', rot_matrix, verts_e2[vert])
            bb2 = np.array([np.dot(aa, faces_e1[i]),
                            np.dot(aa, faces_e2[i])])

            # solve for 2FF at vertex
            FF2_face = np.array([[faces_2FF[i, 0], faces_2FF[i, 1]],
                                 [faces_2FF[i, 1], faces_2FF[i, 2]]])
            e = np.dot(bb1, np.einsum('...ij,...j->...i', FF2_face, bb1))
            f = np.dot(bb2, np.einsum('...ij,...j->...i', FF2_face, bb1))
            g = np.dot(bb2, np.einsum('...ij,...j->...i', FF2_face, bb2))
            verts_2FF[vert] += verts_2FF_weights[i, j] * np.array([[e, f], [f, g]])
            verts_weight_sums[vert] += verts_2FF_weights[i, j]
    verts_2FF /= verts_weight_sums[..., np.newaxis, np.newaxis]
    return verts_2FF


def compute_and_print_outputs(verts, faces, face_areas, verts_2FF):
    """compute and print surface area, volume, and Willmore energy
    Willmore energy is the integral of the squared difference between
    principal curvatures; principal curvatures are the eigenvals of the
    second fundamental form"""
    # surface area
    surface_area = np.sum(face_areas)
    print("Surface Area: ", surface_area)
    # volume
    tri_vols = [np.dot(verts[face[0]],
                np.cross(verts[face[1]], verts[face[2]]))/6.
                for face in faces]
    mesh_vol = abs(np.sum(np.array(tri_vols)))
    print("Mesh Volume: ", mesh_vol)
    # Willmore energy
    vert_principal_curvatures = np.linalg.eigvalsh(verts_2FF)
    willmore_energy = 0.25 * np.sum((vert_principal_curvatures[:, 0] -
                                     vert_principal_curvatures[:, 1])**2)
    print("Willmore energy: ", willmore_energy)
    return vert_principal_curvatures


def write_output(neuropil, verts, faces, vert_principal_curvatures, args):
    """save the surface mesh with pymesh Gaussian curvature and Gaussian
    curvature taken from my own second fundamental forms as a sanity
    check"""
    # construct polydata object with mesh verts and faces
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(verts))
    polydata.SetPoints(points)
    insert_points = range(0, len(faces)*3, 3)
    faces_vtk_format = np.insert(faces.ravel(), insert_points, 3).astype(np.int)
    cells = vtk.vtkCellArray()
    cells.SetCells(faces.shape[0], numpy_to_vtkIdTypeArray(faces_vtk_format))
    polydata.SetPolys(cells)

    # save pymesh and my gaussian curvatures, just as visual inspection
    my_gaussian_curv = np.prod(vert_principal_curvatures, axis=-1)
    polydata.GetPointData().AddArray(numpy_to_vtk(my_gaussian_curv))
    neuropil.add_attribute("vertex_gaussian_curvature")
    pymesh_gaussian_curv = neuropil.get_attribute("vertex_gaussian_curvature")
    polydata.GetPointData().AddArray(numpy_to_vtk(pymesh_gaussian_curv))

    # write output
    polydata.Modified()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(args.mesh_output_path)
    writer.SetInputData(polydata)
    writer.Write()


def main():
    """The main control flow through the script:
    load the mesh, compute and store some reused essentials, compute
    orthonormal coordinate systems at each face and vertex, estimate
    second fundamental forms at each face and vertex, compute outputs,
    write .vtp file to visually inspect results"""
    args = parse_inputs()

    a = load_mesh(args.mesh_input_path)
    neuropil = a[0]
    verts = a[1]
    faces = a[2]

    edges = compute_edge_lengths(verts, faces)

    a = compute_face_properties(verts, faces, edges)
    face_normals = a[0]
    face_areas = a[1]
    faces_e1 = a[2]
    faces_e2 = a[3]

    a = compute_vertex_properties(verts, faces, face_normals, face_areas)
    vert_normals = a[0]
    verts_e1 = a[1]
    verts_e2 = a[2]

    faces_2FF = estimate_2FF_at_faces(faces, edges, faces_e1, faces_e2, vert_normals)
    verts_2FF = estimate_2FF_at_vertices(verts, faces, face_areas, faces_2FF,
                                         face_normals, faces_e1, faces_e2,
                                         vert_normals, verts_e1, verts_e2)

    vert_principal_curvs = compute_and_print_outputs(verts, faces,
                                                     face_areas, verts_2FF)
    write_output(neuropil, verts, faces, vert_principal_curvs, args)


if __name__ == '__main__':
    main()

