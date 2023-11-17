import gmsh
import numpy as np
from mpi4py import MPI

import ufl
from ufl import ds, dx, grad, inner, VectorElement, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction, Measure
import time
from dolfinx import mesh, fem, io, plot, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc

import pyvista

from petsc4py import PETSc

teststart = time.perf_counter()
print("start the simulation")

# Create mesh and define function space
rank = MPI.COMM_WORLD.rank
comm = MPI.COMM_WORLD

timestep = 1e-9
meshsize = 1e-4    
conductiontime = 1e-6
t = 0

L = 0.025
H = 0.005
gdim = 2

# generate the mesh configuration with Delaunay triangles

gmsh.initialize()

if rank == 0:

    rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H) 
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(gdim, [rect], 1)
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.optimize("Netgen")  
    gmsh.write("quenching.msh")   # output the mesh configuration
    
gmsh.finalize() 

# read the mesh configuration from the source file

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
msh, cell_markers, facet_markers = io.gmshio.read_from_msh("quenching.msh", mesh_comm, gmsh_model_rank, gdim=gdim)

# set functions for the FEM variational equation

V = fem.FunctionSpace(msh, ("Lagrange", 1))
x = SpatialCoordinate(msh)
T = TrialFunction(V)
v = TestFunction(V)
T_1 = fem.Function(V)
Th = fem.Function(V)

# set thermal diffusivity and initial conditions

alpha = 300 / 2450 / 0.775
T_1.interpolate(lambda x: 680.0 + 0 * x[0])

# set boundary conditions (Dirichelet and Neumann)

def temp_bc(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0))

dofs_crack = fem.locate_dofs_geometrical(V, temp_bc)
temp_boundary = fem.Function(V)
temp_boundary.interpolate(lambda x: 300.0 + 0 * x[0])
temp_boundary.x.scatter_forward()
bcs = [fem.dirichletbc(temp_boundary, dofs_crack)]

dt = fem.Constant(msh, default_scalar_type(timestep))

# set FEM variational equation

a =  inner(T, v) * dx + dt * alpha * inner(grad(T), grad(v)) * dx 
L = inner(T_1, v) * dx
