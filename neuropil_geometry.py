#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:47:15 2018

@author: fleishmang
"""


import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
import pymesh
import numpy as np
import math


# LOAD MESH AND EXTRACT VERTICIES AND FACES
mesh_path = '/Users/fleishmang/Documents/ModuleNotes/CompMethodsModule'
mesh_path += '/neuropil_blob.obj'
neuropil = pymesh.load_mesh(mesh_path)
verts = neuropil.vertices
faces = neuropil.faces
# END: LOAD MESH AND EXTRACT VERTICIES AND FACES


# COMPUTE EDGE LENGTHS
edges = np.empty((len(faces),) + (3, 3))
for i, face in enumerate(faces):
    a = [verts[face[(ii+1)%3]]-verts[face[ii]] for ii in range(3)]
    edges[i] = np.array(a)
# END: COMPUTE EDGE LENGTHS


# COMPUTE FACE NORMALS, AREAS, AND TANGENT SPACE FRAMES
faces_e1 = edges[:, 0] + 0.5*edges[:, 1]
face_normals = np.cross(faces_e1, edges[:, 1])
face_areas = np.linalg.norm(face_normals, axis=1)/2
face_normals /= face_areas[..., np.newaxis]*2
faces_e1 /= np.linalg.norm(faces_e1, axis=1)[..., np.newaxis]
faces_e2 = np.cross(face_normals, faces_e1)  # ordering ensures r.h. coords
# END: COMPUTE FACE NORMALS, AREAS, AND TANGENT SPACE FRAMES


# COMPUTE VERTEX NORMALS
# weighted average of face normals in 1-ring neighborhood
# weights = face_area/edge1**2/edge2**2 (edge1, edge2 join at vert)
vert_normals = np.zeros_like(verts).astype(np.float32)
for i, face in enumerate(faces):
    weighted_normal = face_normals[i]*face_areas[i]
    for j, vert in enumerate(face):
        # this is doubley inefficient, computing each edge length twice
        vert_wn = weighted_normal/np.sum((verts[face[(j+1)%3]]-verts[vert])**2)
        vert_wn = weighted_normal/np.sum((verts[face[(j+2)%3]]-verts[vert])**2)
        vert_normals[vert] += vert_wn
vert_normals /= np.linalg.norm(vert_normals, axis=1)[..., np.newaxis]
# END: COMPUTE VERTEX NORMALS


# COMPUTE VERTEX TANGENT SPACE FRAMES
verts_e1 = np.ones_like(verts).astype(np.float32)
verts_e1[:, 1] = 0  # let x and y components of e1 be (1, 0)
# uses nx * e1x + ny * e1y + nz * e1z = 0
verts_e1[:, 2] = -vert_normals[:, 0]/vert_normals[:, 2]
verts_e1 /= np.linalg.norm(verts_e1, axis=1)[..., np.newaxis]
verts_e2 = np.cross(vert_normals, verts_e1)
# END: COMPUTE VERTEX TANGENT SPACE FRAMES


# COMPUTE EDGE AND NORMAL DIFFS FOR EACH FACE
# diffs are 2D and represented in the face tangent space frame
edge_diffs = np.empty((len(faces), 6))
norm_diffs = np.empty((len(faces), 6))
for i in range(3):
    edge_diffs[:, 2*i] = np.einsum('...i,...i', edges[:, i], faces_e1)
    edge_diffs[:, 2*i+1] = np.einsum('...i,...i', edges[:, i], faces_e2)
    diff = vert_normals[faces[:, (i+1)%3]] - vert_normals[faces[:, i]]
    norm_diffs[:, 2*i] = np.einsum('...i,...i', diff, faces_e1)
    norm_diffs[:, 2*i+1] = np.einsum('...i,...i', diff, faces_e2)
# END: COMPUTE EDGE AND NORMAL DIFFS FOR EACH FACE


# ESTIMATE 2FF (SECOND FUNDAMENTAL FORM) FOR EACH FACE
# helper function for reformatting edge differences
dummy_matrix = np.zeros((6, 3))
def populate_dummy_matrix(dm, x):
    for i in range(3):
        dm[2*i, :2] = x[2*i:2*i+2]
        dm[2*i+1, 1:3] = x[2*i:2*i+2]
    return dm
# least squares estimate for 2FF from vertex and norms finite diffs
faces_2FF = np.empty_like(faces).astype(np.float32)
for i, face in enumerate(faces):
    dummy_matrix = populate_dummy_matrix(dummy_matrix, edge_diffs[i])
    faces_2FF[i] = np.linalg.lstsq(dummy_matrix, norm_diffs[i], rcond=None)[0]
# END: ESTIMATE FF2 (SECOND FUNDAMENTAL FORM) FOR EACH FACE


# COMPUTE VORONOI BASED WEIGHTS FOR EACH FACE/VERTEX PAIR
verts_2FF_weights = np.zeros_like(faces).astype(np.float32)
vc_temp = []
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
    vc_temp.append(voronoi_center)
    for j in range(3):
        cp = np.cross(voronoi_center - verts[face[j]],
                      midpoints[j] - midpoints[(j-1)%3])
        verts_2FF_weights[i, j] = 0.5 * np.linalg.norm(cp) / face_areas[i]
# END: COMPUTE VORONOI BASED WEIGHTS FOR EACH FACE/VERTEX PAIR


# HELPER FUNCTION FOR ROTATIONS
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
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
# END: HELPER FUNCTION FOR ROTATIONS


# COMPUTE FF2 FOR EACH VERTEX
verts_2FF = np.empty((verts.shape[0],) + (2, 2)).astype(np.float32)
for i, face in enumerate(faces):    
    for j, vert in enumerate(face):
        theta = np.arccos(np.dot(face_normals[i], vert_normals[vert]))
        if np.allclose(theta, 0, atol=1e-03, rtol=1e-04):
            rot_matrix = np.eye(3)
        else:
            rot_axis = np.cross(vert_normals[vert], face_normals[i])
            rot_matrix = rotation_matrix(rot_axis, theta)
        vert_e1_rot = np.einsum('...ij,...j->...i', rot_matrix, verts_e1[vert])
        vert_e2_rot = np.einsum('...ij,...j->...i', rot_matrix, verts_e2[vert])
        vert_e1_proj = np.array([np.dot(vert_e1_rot, faces_e1[i]),
                                 np.dot(vert_e1_rot, faces_e2[i])])
        vert_e2_proj = np.array([np.dot(vert_e2_rot, faces_e1[i]),
                                 np.dot(vert_e2_rot, faces_e2[i])])
        dummy_matrix = np.array([[faces_2FF[i, 0], faces_2FF[i, 1]],
                                 [faces_2FF[i, 1], faces_2FF[i, 2]]])
        e = np.einsum('...ij,...j->...i', dummy_matrix, vert_e1_proj)
        e = np.dot(vert_e1_proj, e)
        f = np.einsum('...ij,...j->...i', dummy_matrix, vert_e1_proj)
        f = np.dot(vert_e2_proj, f)
        g = np.einsum('...ij,...j->...i', dummy_matrix, vert_e2_proj)
        g = np.dot(vert_e2_proj, g)
        verts_2FF[vert] += verts_2FF_weights[i, j] * np.array([[e, f], [f, g]])
# END: COMPUTE FF2 FOR EACH VERTEX
    

# COMPUTE PRINCIPLE CURVATURES AND PRINT WILLMORE ENERGY
vert_principle_curvatures = np.linalg.eigvalsh(verts_2FF)
willmore_energy = 0.25 * np.sum((vert_principle_curvatures[:, 0] - 
                                 vert_principle_curvatures[:, 1])**2)
print("Willmore energy: ", willmore_energy)
# END: COMPUTE PRINCIPLE CURVATURES AND PRINT WILLMORE ENERGY


# PRINT SURFACE AREA
surface_area = np.sum(face_areas)
print("Surface Area: ", surface_area)
# END: PRINT SURFACE AREA


# COMPUTE MESH VOLUME AND PRINT
tri_vols = [np.dot(verts[face[0]], np.cross(verts[face[1]], verts[face[2]]))/6.
            for face in faces]
mesh_vol = abs(np.sum(np.array(tri_vols)))
print("Mesh Volume: ", mesh_vol)
# END: COMPUTE MESH VOLUME AND PRINT
    
    
# WRITE SURFACE AS .VTP
# save vertices
polydata = vtk.vtkPolyData()
points = vtk.vtkPoints()
points.SetData(numpy_to_vtk(verts))
polydata.SetPoints(points)
# save faces
faces_vtk_format = np.insert(faces.ravel(),
                             range(0, len(faces)*3, 3), 3).astype(np.int64)
cells = vtk.vtkCellArray()
cells.SetCells(faces.shape[0], numpy_to_vtkIdTypeArray(faces_vtk_format))
polydata.SetPolys(cells)

# save pymesh and my gaussian curvatures
polydata.GetPointData().AddArray(numpy_to_vtk(np.prod(vert_principle_curvatures, axis=-1)))

neuropil.add_attribute("vertex_gaussian_curvature")
pymesh_check = neuropil.get_attribute("vertex_gaussian_curvature")
polydata.GetPointData().AddArray(numpy_to_vtk(pymesh_check))
# update polydata object and write file
polydata.Modified()
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName('/Users/fleishmang/Desktop/my_output.vtp')
writer.SetInputData(polydata)
writer.Write()
# END: WRITE SURFACE AS .VTP
