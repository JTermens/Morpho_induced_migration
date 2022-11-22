"""

"""

from mshr import *
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

def perturbed_circle_domain_v1(cut_angle, dens_edges, R, axis_angle = 0):
    """
    This function returns a FEniCS mesh constructed from a polynomial approximation
    to a circle with a cut.
        * cut_angle: angle of the cutted arch
        * dens_edges: edges/rad of the polynomial approximation
        * R: radius of the perturbed circle
        * axis_angle: angle of the axis of symmetry.
    Needs the following libraries:
        from fenics import *
        from mshr import Polygon
        import numpy as np
    """

    N = int(dens_edges*(2.*np.pi - cut_angle) + 1) # Number of vertex

    # Vertex angles
    angles = [.5*cut_angle + (2.*np.pi - cut_angle)*i/N for i in np.arange(N)]

    # Correct the effect of rounding to int so the symmetry axis is normal to the cut
    angle_corr = .5*((2.*np.pi - .5*cut_angle)-angles[-1])
    angles += angle_corr

    # Vertex coordinates
    domain = Polygon([Point(R*np.cos(angle + axis_angle), R*np.sin(angle + axis_angle))
                    for angle in angles])
    return domain

def perturbed_circle_domain(cut_angle, R):
    circle = Circle(Point(0,0), R)
    cut = Rectangle(Point(R*(1-np.cos(.5*(np.pi-.5*cut_angle))), -R), Point(R+1, R))
    return circle-cut


def bean_domain(dens_edges, R, n = 3, axis_angle = 0, lobes_angle = np.pi/2):
    """
    This function returns a FEniCS mesh constructed from a polynomial approximation
    to
    R(t) = cos^n(t+axis_angle+(lobes_angle-0.5*pi)) + sin^n(t+axis_angle-(lobes_angle-0.5*pi))
        * dens_edges: edges/rad of the polynomial approximation
        * R: radius
        * n: power of the lobes fucntions
        * axis_angle: angle of the axis of symmetry.
        * lobes_angle: angle between lobes
    Needs the following libraries:
        from fenics import *
        from mshr import Polygon
        import numpy as np
    """

    N = int(dens_edges*np.pi + 1) # Number of vertex

    # Vertex angles
    angles = [np.pi*i/N for i in np.arange(N+1)]

    # Radius of the bean
    radius = lambda angle : np.cos(angle + (axis_angle + .25*np.pi) + .5*(lobes_angle - .5*np.pi))**n \
                          + np.sin(angle + (axis_angle + .25*np.pi) - .5*(lobes_angle - .5*np.pi))**n

    # Vertex coordinates
    domain = Polygon([Point(radius(angle)*np.cos(angle), radius(angle)*np.sin(angle))
                    for angle in angles])
    return domain

if __name__ == '__main__':

    r = 4.
    cut_angle = np.pi/2
    dens_edges = 20

    # Perturbed circle
    domain = perturbed_circle_domain_v1(cut_angle, dens_edges, r)
    mesh = generate_mesh(domain, dens_edges)
    plt.figure()
    plot(mesh, title='Perturbed circle')

    # Perturbed circle v2
    domain = perturbed_circle_domain(cut_angle, r)
    mesh = generate_mesh(domain, dens_edges)
    plt.figure()
    plot(mesh, title='Perturbed circle v2')

    # Bean
    domain = bean_domain(dens_edges, r)
    mesh = generate_mesh(domain, dens_edges)
    plt.figure()
    plot(mesh)

    plt.show()
