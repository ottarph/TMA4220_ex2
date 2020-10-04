import numpy as np
from gauss_quadrature import *

class Femsolver:
    """
        Finite element solver for 1D elliptic pde of the form
            -u_xx + sigma * u = f, on (a,b)
            u(a) = u_1, u(b) = u_2, sigma >= 0
    """

    def __init__(self, sigma, f, a, b, u_1, u_2):

        self.sigma = sigma

        """ Load function """
        self.f = f
        """ Dirichlet boundary conditions """
        self.u_1 = u_1
        self.u_2 = u_2

        """ Definition of boundary """
        self.a = a
        self.b = b


        """------------- Elementary quantities -------------"""
        """ p = 1 """
        self.Ke_1 = np.array([[ 1, -1],
                              [-1,  1]], dtype=float)
        """ p = 2 """
        self.Ke_2 = np.array([[ 7/3, -8/3,  1/3],
                              [-8/3, 16/3, -8/3],
                              [ 1/3, -8/3,  7/3]], dtype=float)


        """------------- Active solver parameters -------------"""

        """ Degree of polynomial interpolant """
        self.p = None
        """ Number of elements """
        self.N = None


        """ Stiffness matrix """
        self.A = None
        """ Load vector """
        self.F = None



    def build_stiffness_matrix(self, p, N):
        pass




def main():

    u_ex = lambda x: np.sin(np.pi * x)
    f = lambda x: np.pi**2 * np.sin(np.pi * x)
    sigma = 0
    a = 0
    b = 2
    u_1 = 0
    u_2 = 0

    femsolver = Femsolver(sigma, f, a, b, u_1, u_2)

    print(femsolver.Ke_1)
    print(femsolver.Ke_2)

    return


if __name__ == '__main__':
    main()
    