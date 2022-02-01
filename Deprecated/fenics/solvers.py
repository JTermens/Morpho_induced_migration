"""

"""

# PDE definition and solver engine
from dolfin import *

# Basic utility
import numpy as np
import matplotlib.pyplot as plt
from lib.utilities import *

def polarity_solver(mesh, **param):
    """
    """
    # Extract the parameters
    h      = param['h']
    T0     = param['T0']
    Lc     = param['Lc']
    zeta   = param['zeta']
    visc   = param['visc']

    # Define a vector function space from a given mesh
    Q = VectorFunctionSpace(mesh, "CG", 2, dim=2) # Polarity

    # Define boundary
    boundary = 'on_boundary'
    # strong imposition of Dirichlet BC for the polarity:
    # p = normal_vector -> g = 1.0
    # This imposition could be done weakly by applying the Nitsche's method,
    # would be worth it?
    N = NormalBoundary(mesh, nor_g=1.0, degree=3)
    bc = DirichletBC(Q, N, boundary)

    # Define trial and test functions
    p = TrialFunction(Q) # Polarity, p
    q = TestFunction(Q)

    # Define functions for the finite element solutions
    ph = Function(Q) # Finite element Polarity

    # Define the variational problem for the polarity
    f_p = Constant((0,0)) # No source of polarity

    c = lambda p, q: -(inner(grad(p), grad(q)) + (1/Lc**2)*dot(p,q))*dx
    source_form = lambda f, q: dot(f,q)*dx

    # Assemble the bilinear forms and apply Dirichlet BC
    Ap = assemble(c(p,q))
    bc.apply(Ap)

    # Solve the polarity equation
    bp = assemble(source_form(f_p, q)) # Assemble a null linear form
    bc.apply(bp) # Apply the Dirichlet BC to it
    solve(Ap, ph.vector(), bp)

    return ph

def velocity_solver(mesh, p, confined=False, **param):
    """

    """
    # Extract the parameters
    h      = param['h']
    T0     = param['T0']
    Lc     = param['Lc']
    zeta   = param['zeta']
    visc   = param['visc']

    # Define a vector function space from a given mesh
    V = VectorFunctionSpace(mesh, "CG", 2, dim=2) # Velocity
    if confined: # If the tissue is confined, the velocity at the edges is 0
        boundary = 'on_boundary'
        bc = DirichletBC(Q, Constant((0,0)), boundary)

    # Define trial and test functions
    u = TrialFunction(V) # Velocity, u
    v = TestFunction(V)

    # Define functions for the finite element solutions
    uh = Function(V) # Finite element Velocity

    # Define the variational problem for the velocity
    a = lambda u, v: - h*visc*inner(sym(grad(u)), sym(grad(v)))*dx
    b = lambda p, v: (h*zeta*inner(outer(p, p), sym(grad(v))) + T0*dot(p,v))*dx

    # Assemble the bilinear forms
    Au = assemble(a(u,v))

    # Solve the force balance equation
    bu = assemble(-b(p,v)) # Assemble the linear form
    if confined:
        bc.apply(Au)
        bc.apply(bu)

    solve(Au, uh.vector(), bu)

    return uh

def velocity_solver_v2(mesh, p, **param):
    """

    """
    # Extract the parameters
    h      = param['h']
    T0     = param['T0']
    Lc     = param['Lc']
    zeta   = param['zeta']
    visc   = param['visc']

    # Define a vector function space from a given mesh
    V = VectorFunctionSpace(mesh, "CG", 2, dim=2) # Velocity

    # Define trial and test functions
    u = TrialFunction(V) # Velocity, u
    v = TestFunction(V)

    # Define functions for the finite element solutions
    uh = Function(V) # Finite element Velocity

    # High-order space to ensure source smoothness
    W = VectorFunctionSpace(mesh, "CG", 4, dim=2)

    # Define the source and bc terms
    source = project((-h*zeta*div(outer(p, p)) + T0*p), W)
    n = FacetNormal(mesh) # Normal vector to the boundary
    bc = zeta*h*n

    # Define the variational problem for the velocity
    a = lambda u, v: - h*visc*inner(sym(grad(u)), sym(grad(v)))*dx
    b = lambda source, v: dot(source, v)*dx
    surf = lambda bc, v: dot(bc, v)*ds

    # Assemble the bilinear forms
    Au = assemble(a(u,v))

    # Solve the force balance equation
    bu = assemble(-b(source,v) - surf(bc,v)) # Assemble the linear form
    solve(Au, uh.vector(), bu)

    return uh


if __name__ == '__main__':
    # minimal example

    # Model parameters, lenghts are expressed in μm and pressures in kPa
    h      =  5.0  # Monolayer height, μm
    T0     =  0.5  # Maximal traction, kPa, 0.2-0.8
    Lc     =  25.0 # Nematic lenght, μm
    zeta   = -20.0 # zeta = - Contrtactility, kPa, 5-50
    visc   =  50e3 # Monolayer viscosity, kPa·s, 3e3-3e4
    R_crit = .5*(3*Lc - zeta*h/T0) # Critical radius

    # Domain parameters
    radius =  200  # Tissue radius
    mesh_resolution = 60

    # Define a domain and generate a mesh
    domain = Circle(Point(0, 0), radius)
    mesh = generate_mesh(domain, mesh_resolution)

    # Solve the polarity equation
    p = polarity_solver(mesh, h=h, T0=T0, Lc=Lc, zeta=zeta, visc=visc)

    # Solve the force balance equation
    u = velocity_solver(mesh, p, h=h, T0=T0, Lc=Lc, zeta=zeta, visc=visc)

    plt.figure()
    p_plot = plot(p, title='Polarity, $p$')
    cbar = plt.colorbar(p_plot)

    plt.figure()
    u_plot = plot(u, title='Velocity, $u$')
    cbar = plt.colorbar(u_plot)

    plt.show()
