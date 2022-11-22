# PDE definition and solver engine
from dolfin import *
from solvers import *

# Basic utility
import numpy as np
import matplotlib.pyplot as plt
from lib.utilities import *

# Domains and mesh generation
from mshr import *
from lib.domains import perturbed_circle_domain

# Model parameters,
# lenghts are expressed in μm, pressures in kPa and time in s.
h      =  5.0  # Monolayer height, μm
T0     =  0.5  # Maximal traction, kPa, 0.2-0.8
Lc     =  25.0 # Nematic lenght, μm
zeta   = -20.0 # zeta = - Contrtactility, kPa, 5-50
visc   =  50e3 # Monolayer viscosity, kPa·s, 3e3-3e4
R_crit = .5*(3*Lc - zeta*h/T0) # Critical radius

# Domain parameters
radius =  200  # Tissue radius, μm
mesh_resolution = 60

# Define the expressions for the analytical (exact) solutions
p_e, u_e = circular_solutions(degree=4, h=h, T0=T0, Lc=Lc, zeta=zeta,
                                 visc=visc, radius = radius)

# Define a circular domain and generate a mesh
domain = Circle(Point(0, 0), radius)
mesh = generate_mesh(domain, mesh_resolution)

# Solve the polarity equation
p = polarity_solver(mesh, h=h, T0=T0, Lc=Lc, zeta=zeta, visc=visc)

error_p = errornorm(p_e, p, norm_type='L2')
print('L2 error for the polarity = {:.4E}'.format(error_p))

plt.figure()
p_e_plot = plot(p_e, mesh=mesh, title='Analytical Polarity')
cbar = plt.colorbar(p_e_plot)
cbar.ax.set_ylabel('$|p|$')
plt.xlabel('x [$\mu$m]')
plt.ylabel('y [$\mu$m]')

plt.figure()
p_plot = plot(p, title='Computed Polarity\nL2 error = {:.4E}'.format(error_p))
cbar = plt.colorbar(p_plot)
cbar.ax.set_ylabel('$|p|$')
plt.xlabel('x [$\mu$m]')
plt.ylabel('y [$\mu$m]')


## Try different implementations of the polarity field to the force balance eq
# Polarity obtained from solving the polarity eq
u_p = velocity_solver(mesh, p, h=h, T0=T0, Lc=Lc, zeta=zeta, visc=visc)
error_u_p = errornorm(u_e, u_p, norm_type='L2')
print('L2 error for u_p = {:.4E}'.format(error_u_p))

# Exact Polarity
u_p_e = velocity_solver(mesh, p_e, h=h, T0=T0, Lc=Lc, zeta=zeta, visc=visc)
error_u_p_e = errornorm(u_e, u_p_e, norm_type='L2')
print('L2 error for u_p_e = {:.4E}'.format(error_u_p_e))

plt.figure()
u_e_plot = plot(u_e, mesh=mesh, title='Analytical velocity')
cbar = plt.colorbar(u_e_plot)
cbar.ax.set_ylabel('$|v|$ [$\mu$m/s]')
plt.xlabel('x [$\mu$m]')
plt.ylabel('y [$\mu$m]')

plt.figure()
u_p_plot = plot(u_p, title='Velocity from computed polarity\nL2 = {:.4E}'.format(error_u_p))
cbar = plt.colorbar(u_p_plot)
cbar.ax.set_ylabel('$|v|$ [$\mu$m/s]')
plt.xlabel('x [$\mu$m]')
plt.ylabel('y [$\mu$m]')

plt.figure()
u_p_e_plot = plot(u_p_e, title='Velocity from analytical polarity\nL2 = {:.4E}'.format(error_u_p_e))
cbar = plt.colorbar(u_p_e_plot)
cbar.ax.set_ylabel('$|v|$ [$\mu$m/s]')
plt.xlabel('x [$\mu$m]')
plt.ylabel('y [$\mu$m]')

plt.show()
