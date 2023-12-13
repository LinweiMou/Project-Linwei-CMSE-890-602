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


def read_paper_information(file_path):
    """
    Reads information from a text file and extracts key-value pairs.

    Parameters:
    - file_path (str): The path to the text file containing information.

    Returns:
    - data (dict): A dictionary containing the extracted key-value pairs.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize an empty dictionary to store key-value pairs
    data = {}

    # Loop through each line in the file
    for line in lines:
        # Strip leading and trailing whitespaces from the line
        line = line.strip()

        # Skip lines starting with '#' (comments) or empty lines
        if line.startswith('#') or not line:
            continue

        # Split the line into key and value based on '='
        key, value_str = [item.strip() for item in line.split('=')]
        key = key.strip()

        # Check if the value is a list or a single value
        if '[' in value_str and ']' in value_str:
            # Parse the list
            value = [item.strip(" []") for item in value_str.split(',')]
            # Convert numeric values to float, excluding non-numeric values
            value = [float(item) if (item.replace('.', '', 1).isdigit() or 'e' in item.lower()) and not item.isalpha() else item for item in value]
        else:
            # Parse the single value
            # Convert numeric values to float, excluding non-numeric values
            value = float(value_str) if (value_str.replace('.', '', 1).isdigit() or 'e' in value_str.lower()) and not value_str.isalpha() else value_str

        # Store the key-value pair in the dictionary
        data[key] = value

    return data

# read data from paper information
file_path = "paper_information.txt"
paper_data = read_paper_information(file_path)


# access the information using the keys
# print("Dimension:", paper_data['dimension'])
# print("Boundary Condition:", paper_data['boundary_condition'])
# print("Initial Condition:", paper_data['initial_condition'])
# print("Time Range:", paper_data['time_range'])
# print("Properties:", paper_data['properties'])


timestep = 1e-9
meshsize = 2e-4    
conductiontime = 1e-7
t = 0
t_cache = 0
storagetime = 10

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


bilinear_form = fem.form(a)
linear_form = fem.form(L)
A = fem.petsc.create_matrix(bilinear_form)
b = fem.petsc.create_vector(linear_form)

solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.GMRES)
solver.getPC().setType(PETSc.PC.Type.JACOBI)
solver.setTolerances(rtol=1e-9)
solver.setTolerances(atol=1e-13)
solver.setTolerances(max_it=1000)


while (t<conductiontime):
    
    t += timestep

    A.zeroEntries()
    fem.petsc.assemble_matrix(A, bilinear_form, bcs=bcs)
    A.assemble()
    with b.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b, linear_form)
    fem.petsc.apply_lifting(b, [bilinear_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    solver.solve(b, Th.vector)
    Th.x.scatter_forward()

    T_1.x.array[:] = Th.x.array
    T_1.x.scatter_forward()    
    
    
    if round(t, 3) % storagetime == 0 and round(t, 3) - t_cache > 5*timestep:
        
        t_cache = round(t, 3)
        Th_cache = Th

        if rank == 0:
                
            alldata_out = {'Th': Th_cache}
            alldata_out = pd.DataFrame.from_dict(alldata_out, orient='index')
            alldata_out = alldata_out.transpose()
            data_write_path = r"temp@" + str(round(t)) + "s.csv"      
            alldata_out.to_csv(data_write_path, index=False, mode='w') 


topo, types, geom = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topo, types, geom)
grid.point_data["temperature"] = Th.x.array.real
grid.set_active_scalars("temperature")
plotter = pyvista.Plotter()
plotter.add_text("temperature", position="upper_edge", font_size=14, color="black")
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()
plotter.show()












