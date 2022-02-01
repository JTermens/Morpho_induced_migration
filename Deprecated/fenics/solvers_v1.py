"""

"""

# from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
from lib.domains import perturbed_circle_domain
from lib.utilities import BoundaryExpression2D

from sys import exit

# Model parameters, lenghts are expressed in μm and pressures in kPa
h      =  5.0  # Monolayer height, μm
T0     =  0.5  # Maximal traction, kPa, 0.2-0.8
Lc     =  25.0 # Nematic lenght, μm
zeta   = -20.0 # zeta = - Contrtactility, kPa, 5-50
visc   =  50e3 # Monolayer viscosity, kPa·s, 3e3-3e4
R_crit = .5*(3*Lc - zeta*h/T0) # Critical radius
radius =  200  # Tissue radius

# Define strain-rate tensor
def epsilon(u):
    return sym(grad(u))

# Define the symmetric stress tensor
def sigma(u, p):
    return (visc*epsilon(u) - zeta*outer(p, p))


def one_step_split_solver(mesh):
    """

    """
    # Define a two vector function spaces from a given mesh
    Q = VectorFunctionSpace(mesh, "CG", 2, dim=2) # Polarity
    V = VectorFunctionSpace(mesh, "CG", 2, dim=2) # Velocity
    U = VectorFunctionSpace(mesh, "CG", 4, dim=2) # Polarity into Velocity

    # Define boundary
    boundary = 'on_boundary'
    # strong imposition of Dirichlet BC for the polarity:
    # p = normal_vector -> g = 1.0
    # This imposition could be done weakly by applying the Nitsche's method,
    # would be worth it?
    n = FacetNormal(mesh)
    N = BoundaryExpression(mesh, nor_g=1.0, degree=2)
    bc = DirichletBC(Q, N, boundary)

    # Define trial and test functions
    p = TrialFunction(Q) # Polarity, p
    q = TestFunction(Q)
    u = TrialFunction(V) # Velocity, v
    v = TestFunction(V)

    # Define functions for the finite element solutions
    ph = Function(Q) # Finite element Polarity
    uh = Function(V) # Finite element Velocity

    # Define the variational problem for the polarity
    f_p = Constant((0,0)) # No source of polarity

    Fp = (1/Lc**2)*dot(p, q)*dx      \
        + inner(grad(p), grad(q))*dx \
        + dot(f_p, q)*dx
    ap, Lp = lhs(Fp), rhs(Fp)

    # Assemble the bilinear forms and apply Dirichlet BC
    Ap = assemble(ap)
    bc.apply(Ap)

    # Step 1: Solve the polarity equation
    bp = assemble(Lp) # Assemble the linear form
    bc.apply(bp) # Apply the Dirichlet BC to it
    solve(Ap, ph.vector(), bp)

    pu = interpolate(ph, U)

    # Define the variational problem for the polarity
    Fu = - inner(sigma(u, pu), epsilon(v))*dx \
         + T0*dot(pu, v)*dx
    au, Lu = lhs(Fu), rhs(Fu)

    Au = assemble(au)

    # Step 2: Solve the force balance equation
    bu = assemble(Lu)
    solve(Au, uh.vector(), bu)

    return ph, uh

def one_step_mono_solver(mesh): # Does not work, needs fixing
    """

    """

    # Define a two vector function spaces from a given mesh
    elem_p = VectorElement("CG", mesh.ufl_cell(), 3, dim=2)
    elem_u = VectorElement("CG", mesh.ufl_cell(), 3, dim=2)

    # Define the mixed space
    W = FunctionSpace(mesh, MixedElement([elem_p, elem_u]))

    # Define boundary
    boundary = 'on_boundary'
    # strong imposition of Dirichlet BC for the polarity:
    # p = normal_vector -> g = 1.0
    # This imposition could be done weakly by applying the Nitsche's method,
    # would be worth it?
    n = FacetNormal(mesh)
    Np = BoundaryExpression(mesh, nor_g=1.0, degree=3)
    bc = DirichletBC(W.sub(0), Np, boundary)

    # Define trial and test functions
    p, u = TrialFunctions(W)
    q, v = TestFunctions(W)

    # Define function for the finite element solution
    wh = Function(W)

    # Define the variational problem
    f_p = f_u = Constant((0,0)) # No source of polarity or velocity

    F = (1/Lc**2)*dot(p, q)*dx + inner(grad(p), grad(q))*dx  + dot(f_p, q)*dx \
      + T0*dot(p, v)*dx - inner(visc*sym(grad(u)), sym(grad(v)))*dx  \
      + inner(zeta*outer(p,p), sym(grad(v)))*dx + dot(f_u, v)*dx

    A, L = lhs(F), rhs(F)

    solve(A==L, wh, bc)

    (ph, uh) = wh.split() # split with shallow copy

    return ph, uh


if __name__ == '__main__':

    # Parameters of the domain
    mesh_resolution = 60

    # Define a domain and generate a mesh
    domain = Circle(Point(0, 0), radius)
    mesh = generate_mesh(domain, mesh_resolution)

    # Splitting solver
    # p_split, v_split = one_step_split_solver(mesh)

    # Monolithic solver
    p_mono, v_mono = one_step_mono_solver(mesh)

    plt.figure()
    plot(p_mono, title='Polarity monolithic, $p$')
    plt.figure()
    plot(v_mono, title='Velocity monolithic, $v$')

    # vtkfile_p = File('circular/polarity.pvd')
    # vtkfile_p << p
    # vtkfile_v = File('circular/velocity.pvd')
    # vtkfile_v << v

    plt.show()
