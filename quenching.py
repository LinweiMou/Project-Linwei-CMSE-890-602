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


# set the parameters for parallel computing
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

# access the information using the keys
# print("Dimension:", paper_data['dimension'])
# print("Boundary Condition:", paper_data['boundary_condition'])
# print("Initial Condition:", paper_data['initial_condition'])
# print("Time Range:", paper_data['time_range'])
# print("Properties:", paper_data['properties'])

def save_paper_information(paper_data):
    """
    Save paper information to multiple text files.

    Parameters:
    - paper_data (dict): A dictionary containing paper information.

    The function saves different aspects of the paper information to separate text files.
    The keys in the 'paper_data' dictionary correspond to different aspects of the paper.

    File 1: 'dimension.txt' - Dimensions of the paper.
    File 2: 'boundary_condition.txt' - Boundary conditions of the paper.
    File 3: 'initial_condition.txt' - Initial conditions of the paper.
    File 4: 'time_range.txt' - Time range of the paper.
    File 5: 'properties.txt' - Properties of the paper.
    """
    
    # File paths for different aspects of paper information
    file1 = "dimension.txt"
    file2 = "boundary_condition.txt"
    file3 = "initial_condition.txt"
    file4 = "time_range.txt"
    file5 = "properties.txt"

    # Save dimensions to 'dimension.txt'
    with open(file1, 'w') as file:
        # Append data to the file
        file.write(str(paper_data['dimension']))

    # Save boundary conditions to 'boundary_condition.txt'
    with open(file2, 'w') as file:
        # Append data to the file
        file.write(str(paper_data['boundary_condition']))

    # Save initial conditions to 'initial_condition.txt'
    with open(file3, 'w') as file:
        # Append data to the file
        file.write(str(paper_data['initial_condition']))

    # Save time range to 'time_range.txt'
    with open(file4, 'w') as file:
        # Append data to the file
        file.write(str(paper_data['time_range']))

    # Save properties to 'properties.txt'
    with open(file5, 'w') as file:
        # Append data to the file
        file.write(str(paper_data['properties']))

def generate_mesh(dimension_info, output_file="quenching_mesh.msh"):
    """
    Generates a mesh using Gmsh based on the provided dimension information.

    Parameters:
    - dimension_info (list): List containing dimension information [L, H, meshsize, gdim].
    - output_file (str): Name of the output mesh file (default is "quenching_mesh.msh").
    """
    # Initialize Gmsh
    gmsh.initialize()

    # Check if the rank is 0 (assuming MPI-like parallelization)
    if rank == 0:
        # Extract dimension information from the input list
        L, H, meshsize, gdim = dimension_info[1], dimension_info[3], dimension_info[5], int(dimension_info[7])

        # Add a rectangle to the Gmsh model
        rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        
        # Synchronize the model
        gmsh.model.occ.synchronize()

        # Add a physical group for the specified dimension
        gmsh.model.addPhysicalGroup(gdim, [rect], 1)

        # Set meshing options
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize)
        gmsh.option.setNumber("Mesh.Algorithm", 5)

        # Generate the mesh
        gmsh.model.mesh.generate(gdim)
        
        # Set the order of the mesh elements
        gmsh.model.mesh.setOrder(1)

        # Optimize the mesh using Netgen
        gmsh.model.mesh.optimize("Netgen")

        # Write the mesh configuration to the specified output file
        gmsh.write(output_file)

    # Finalize Gmsh
    gmsh.finalize()
    

def create_function_space_and_fields(mesh):
    """
    Create a function space and fields for the finite element method.

    Parameters:
    - mesh (Mesh): The mesh for the finite element method.

    Returns:
    - V (FunctionSpace): The function space.
    - x (SpatialCoordinate): The spatial coordinate.
    - T (TrialFunction): The trial function.
    - v (TestFunction): The test function.
    - T_1 (Function): The function T_1.
    - Th (Function): The function Th.
    """

    # Create a function space
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))

    # Define spatial coordinates
    x = SpatialCoordinate(msh)

    # Define trial and test functions
    T = TrialFunction(V)
    v = TestFunction(V)

    # Initialize functions
    T_1 = fem.Function(V)
    Th = fem.Function(V)

    # Interpolate initial condition for T_1
    T_1.interpolate(lambda x: 680.0 + 0 * x[0])

    return V, x, T, v, T_1, Th


def temp_bc(x):
    """
    Define a boolean function to represent a temperature boundary condition.

    Parameters:
    - x (numpy.ndarray): A numpy array representing the spatial coordinates [x, y].

    Returns:
    - bc_mask (numpy.ndarray): A boolean array indicating whether the point is on the boundary.
    """
    # Check if x[0] is close to 0 or x[1] is close to 0 (left and bottom)
    bc_mask = np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0))

    return bc_mask


def apply_temp_boundary_condition(V, temp_bc):
    """
    Apply a temperature boundary condition to a given function space.

    Parameters:
    - V (FunctionSpace): The function space for the finite element method.
    - temp_bc (callable): A function that determines the points on the boundary.

    Returns:
    - temp_boundary (Function): The function representing the applied temperature boundary condition.
    - bcs (list): A list of DirichletBC objects specifying the boundary conditions.
    """
    # Locate degrees of freedom on the boundary defined by temp_bc
    dofs_crack = fem.locate_dofs_geometrical(V, temp_bc)

    # Create a function representing the temperature boundary condition
    temp_boundary = fem.Function(V)
    temp_boundary.interpolate(lambda x: 300.0 + 0 * x[0])

    # Scatter the values to apply the boundary condition
    temp_boundary.x.scatter_forward()

    # Create a Dirichlet boundary condition using the located degrees of freedom
    bcs = [fem.dirichletbc(temp_boundary, dofs_crack)]

    return temp_boundary, bcs
    
    
def create_solver(bilinear_form, linear_form, mesh_comm):
    """
    Create and configure a PETSc KSP solver for a given bilinear and linear form.

    Parameters:
    - bilinear_form (a bilinear form): The bilinear form of the variational problem.
    - linear_form (a linear form): The linear form of the variational problem.
    - mesh_comm (MPI communicator): The MPI communicator associated with the mesh.

    Returns:
    - solver (PETSc.KSP): The configured PETSc KSP solver.
    - A (PETSc.Mat): The PETSc matrix corresponding to the bilinear form.
    - b (PETSc.Vec): The PETSc vector corresponding to the linear form.
    """
    # Create PETSc matrix and vector for the bilinear and linear forms
    A = fem.petsc.create_matrix(bilinear_form)
    b = fem.petsc.create_vector(linear_form)

    # Create PETSc KSP solver
    solver = PETSc.KSP().create(mesh_comm)
    solver.setOperators(A)

    # Configure the solver
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.JACOBI)
    solver.setTolerances(rtol=1e-9, atol=1e-13, max_it=1000)

    return solver, A, b


def solve_linear_system(A, b, bilinear_form, linear_form, bcs):
    """
    Solve a linear system using a PETSc KSP solver.

    Parameters:
    - A (PETSc.Mat): PETSc matrix corresponding to the bilinear form.
    - b (PETSc.Vec): PETSc vector corresponding to the linear form.
    - bilinear_form (a bilinear form): The bilinear form of the variational problem.
    - linear_form (a linear form): The linear form of the variational problem.
    - bcs (list): A list of DirichletBC objects specifying the boundary conditions.

    Returns:
    - Th (Function): The solution function.
    """
    # Zero out the matrix entries
    A.zeroEntries()

    # Assemble the matrix and apply boundary conditions
    fem.petsc.assemble_matrix(A, bilinear_form, bcs=bcs)
    A.assemble()

    # Zero out the local vector
    with b.localForm() as loc_b:
        loc_b.set(0)

    # Assemble the vector and apply boundary conditions
    fem.petsc.assemble_vector(b, linear_form)
    fem.petsc.apply_lifting(b, [bilinear_form], [bcs])

    # Update the ghost values in the vector
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Apply boundary conditions to the vector
    fem.petsc.set_bc(b, bcs)

   

    # Solve the linear system
    solver.solve(b, Th.vector)

    # Update the ghost values in the solution vector
    Th.x.scatter_forward()
    
    # Update the values to next timestep and the ghost values
    T_1.x.array[:] = Th.x.array
    T_1.x.scatter_forward() 

    return Th
    

def save_solution_plot(V, Th, output_file="solution_plot.png"):
    """
    Save a temperature distribution plot to a file.

    Parameters:
    - V (FunctionSpace): The function space.
    - Th (Function): The solution function.
    - output_file (str): Name of the output file (default is "solution_plot.png").
    """
    topo, types, geom = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topo, types, geom)
    grid.point_data["temperature"] = Th.x.array.real
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_text("Temperature Distribution", position="upper_edge", font_size=14, color="black")
    plotter.add_mesh(grid, show_edges=False)
    plotter.view_xy()

    # Save the plot to a file
    plotter.show(screenshot=output_file)


# the main script starts here

# read data from paper information
file_path = "paper_information.txt"
paper_data = read_paper_information(file_path)
save_paper_information(paper_data)

# generate the quenching mesh
dimension_info = paper_data['dimension']
generate_mesh(dimension_info)

# read the mesh configuration from the generated mesh
gmsh_model_rank = 0
msh, cell_markers, facet_markers = io.gmshio.read_from_msh("quenching_mesh.msh", comm, gmsh_model_rank, gdim=2)

# generate function space and fields
V, x, T, v, T_1, Th = create_function_space_and_fields(msh)

# apply boundary conditions
temp_boundary, bcs = apply_temp_boundary_condition(V, temp_bc)

# set time step and time range
timestep, conductiontime = paper_data['time_range'][1], paper_data['time_range'][3]
dt = fem.Constant(msh, default_scalar_type(timestep))
t = 0

# set parameters
alpha = paper_data['properties'][1]

# set FEM variational equation and corresponding solver
a =  inner(T, v) * dx + dt * alpha * inner(grad(T), grad(v)) * dx 
L = inner(T_1, v) * dx
bilinear_form = fem.form(a)
linear_form = fem.form(L)
solver, A, b = create_solver(bilinear_form, linear_form, msh.comm)


while (t<conductiontime):
    
    t += timestep

    Th = solve_linear_system(A, b, bilinear_form, linear_form, bcs)
    
    if rank == 0:
        print("time step:", round(t, 9))    
    
    
save_solution_plot(V, Th, output_file="my_result.png")












