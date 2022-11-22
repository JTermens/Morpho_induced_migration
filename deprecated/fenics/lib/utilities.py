"""

"""

from dolfin import *
from mshr import Circle, generate_mesh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt

# Define an expression that allow to determine boundaries from its
# local components (normal and tangent)
class NormalBoundary(UserExpression):
    def __init__(self, mesh, nor_g=0, tan_g=0, **kwargs):
        self.mesh = mesh
        self.nor_g = nor_g
        self.tan_g = tan_g
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        values[0] = self.nor_g*n[0] - self.tan_g*n[1]
        values[1] = self.nor_g*n[1] + self.tan_g*n[0]
    def value_shape(self):
        return (2,)

# Define an expression that returns the values of a given
# function
class FunctionExpression(UserExpression):
    def __init__(self, fun, **kwargs):
        self.fun = fun
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = self.fun(x[0], x[1])[0]
        values[1] = self.fun(x[0], x[1])[1]

    def value_shape(self):
        return (2,)


def get_normal_boundary(mesh, degree=1):
    """
    Returns the normal vector to the mesh boundary as a dolfin
    function. Arguments:
        * mesh: mshr mesh
        * degree: degree of the Continuous Galerkin (CG) elements
        used to approximate the normal. By default 1.
    """
    n = FacetNormal(mesh)
    V = VectorFunctionSpace(mesh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u,v)*ds
    l = inner(n, v)*ds
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nh = Function(V)

    solve(A, nh.vector(), L)
    return nh

def matrix_logdet(matrix):
    """
    Compute the log determinant of a dolfin.cpp.la.Matrix object.
    To do so, the matrix object is converted to a sparse matrix
    and its determinant is approximated by the product of the
    determinants of the triangular matrices of an M=LU decomposition.
    Arguments:
        * matrix: a dolfin.cpp.la.Matrix object
    """

    # Convert the matrix object to a sparce matrix in csr
    mat = as_backend_type(matrix).mat()
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)

    # LU decomposition of the sparse matrix
    lu = splu(csr.tocsc())

    # The determinant of triangular matrices is the product
    # of the diagonal terms
    diagL = lu.L.diagonal()
    diagU = lu.U.diagonal()

    # Invoke complex arithmetic to account for negative numbers
    # that might appear in the diagonals
    diagL = diagL.astype(np.complex128)
    diagU = diagU.astype(np.complex128)

    # Compute the log(det) to avoid over/underflow
    logdet = np.log(diagL).sum() + np.log(diagU).sum()

    return logdet

def circular_solutions(degree=3, **param):
    """
    This function returns the analytical solutions to the
    polarity and force balance equations for a circular domain
    as UFL Expressions with a given degree (3 by default). Arguments:
        * degree: degree of the UFL expression. 3 by default
        * param: dictionary with the model parameters
    """

    from scipy.special import iv

    h      = param['h']
    T0     = param['T0']
    Lc     = param['Lc']
    zeta   = param['zeta']
    visc   = param['visc']
    R = param['radius']

    radial_p = lambda r: iv(1, r/Lc)/iv(1, R/Lc)
    radial_u = lambda r: 1/(2*visc)*((zeta - 2*T0*Lc**2/(h*R) \
             + (zeta*Lc/R + 2*T0*Lc/h)*iv(0, R/Lc)/iv(1, R/Lc)  \
             - zeta*iv(0, R/Lc)**2/iv(1, R/Lc)**2)*r + (zeta*iv(0, r/Lc)/iv(1, R/Lc) \
             - 2*T0*Lc/h)*Lc*iv(1, r/Lc)/iv(1, R/Lc))

    p = lambda x, y: [radial_p(np.sqrt(x**2 + y**2)) * np.cos(np.arctan2(y, x)), \
                      radial_p(np.sqrt(x**2 + y**2)) * np.sin(np.arctan2(y, x))]
    u = lambda x, y: [radial_u(np.sqrt(x**2 + y**2)) * np.cos(np.arctan2(y, x)), \
                      radial_u(np.sqrt(x**2 + y**2)) * np.sin(np.arctan2(y, x))]

    return FunctionExpression(p, degree=degree), FunctionExpression(u, degree=degree)


if __name__ == '__main__':
    # Minimal example
    radius = 200
    mesh_resolution = 60

    domain = Circle(Point(0, 0), radius)
    mesh = generate_mesh(domain, mesh_resolution)
    V = FunctionSpace(mesh, 'CG', 1)
    u, v = TrialFunction(V), TestFunction(V)

    A = assemble(inner(u, v)*dx)

    # Compute matrix determinant
    logdet = matrix_logdet(A)
    print(logdet)

    p_sol, u_sol = circular_solutions(  \
                        h      =  5.0,  \
                        T0     =  0.5,  \
                        Lc     =  25.0, \
                        zeta   = -20.0, \
                        visc   =  50e3, \
                        radius =  200,  \
                        )

    # Get and plot the normal boundary field
    n = get_normal_boundary(mesh)
    plt.figure()
    plot(mesh)
    plot(n)

    plt.figure()
    p_plot = plot(p_sol, mesh = mesh, title = "Analytical polarity", mode = "glyphs")
    cbar = plt.colorbar(p_plot)

    plt.figure()
    u_plot = plot(u_sol, mesh = mesh, title = "Analytical velocity")
    cbar = plt.colorbar(u_plot)

    plt.show()
