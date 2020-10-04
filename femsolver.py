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


        """------------- Active solver variables -------------"""

        """ Degree of polynomial interpolant """
        self.p = None
        """ Number of nodes """
        self.N = None
        """ Number of elements """
        self.K = None

        """ Gauss-Legendre quadrature sample points, set for now. """
        self.quad_samples = 5

        """ Triangulation """
        self.nodes = None
        self.edge_nodes = None
        self.elements = None
        self.h = None


        """ Stiffness matrix """
        self.A = None
        """ Load vector """
        self.F = None
        """ Solution vector """
        self.u = None

        return


    def build_triangulation(self, N):
        """ N+2 nodes, the edge nodes x_0, x_N+1 and N internal nodes x_1,...,x_N """
        self.nodes = np.linspace(self.a, self.b, N+2)
        self.edge_nodes = [self.nodes[0], self.nodes[-1]]
        """ For 1D, K = N+1 elements in the triangulation """
        self.elements = [[self.nodes[i-1], self.nodes[i]] for i in range(1, N+2)]

        self.h = self.nodes[1:] - self.nodes[:-1]

        self.N = N
        self.K = N+1

        return


    def build_stiffness_matrix(self, p):

        if self.elements == None:
            raise Exception("Require triangulation before building stiffness matrix.")
        
        self.p = p

        self.A = np.zeros((self.N+2,self.N+2), dtype=float)

        if p == 1:
            Ke = self.Ke_1

            """ For each element in triangulation """
            for i in range(self.K):
                self.A[i:i+2,i:i+2] += self.h[i] * Ke
            
            self.A = self.A[1:-1,1:-1]

        elif p == 2:
            raise NotImplementedError
        
        return


    def build_load_vector(self, p):
        
        if self.elements == None:
            raise Exception("Require triangulation before building stiffness matrix.")
        
        self.F = np.zeros(self.N+2)

        if p == 1:

            """ For each element in triangulation """
            for i in range(self.K):
                x0, x1 = self.elements[i]
                hi = self.h[i]
                g = lambda x: self.f(x0 + hi * x)
                
                Me1 = gauss_quad(lambda x: (1 - x) * g(x), 0, 1, self.quad_samples)
                Me2 = gauss_quad(lambda x: x * g(x), 0, 1, self.quad_samples)
                Me = np.array([Me1, Me2], dtype=float)

                self.F[i:i+2] += hi * Me

            self.F[1] -= self.h[0] * self.u_1
            self.F[self.N] -= self.h[self.N] * self.u_2

            self.F = self.F[1:-1]

        elif p == 2:
            raise NotImplementedError

        return

    def solve_linear_system(self):
        
        self.u = np.linalg.solve(self.A, self.F)

        return

    def plot_solution(self):







def main():

    u_ex = lambda x: np.sin(np.pi * x)
    f = lambda x: np.pi**2 * np.sin(np.pi * x)
    sigma = 0
    a = 0
    b = 2
    u_1 = 0
    u_2 = 0

    femsolver = Femsolver(sigma, f, a, b, u_1, u_2)

    N = 4
    p = 1
    femsolver.build_triangulation(N)
    femsolver.build_stiffness_matrix(p)
    print(femsolver.A)
    femsolver.build_load_vector(p)
    print(femsolver.F)
    femsolver.solve_linear_system()
    print(femsolver.u)

    return


if __name__ == '__main__':
    main()
    