import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
import matplotlib.pyplot as plt
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

        self.Le_1 = np.array([[1/3, 1/6],
                              [1/6, 1/3]], dtype=float)

        """ p = 2 """
        self.Ke_2 = np.array([[ 7/3, -8/3,  1/3],
                              [-8/3, 16/3, -8/3],
                              [ 1/3, -8/3,  7/3]], dtype=float)
            
        self.Le_2 = np.array([[ 2/15, 1/15, -1/30],
                              [ 1/15, 8/15,  1/15],
                              [-1/30, 1/15,  2/15]], dtype=float)


        """------------- Active solver variables -------------"""

        """ Degree of polynomial interpolant """
        self.p = None
        """ Number of elements """
        self.N = None

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


    def build_triangulation(self, N, p):

        self.p = p

        if p == 1:

            """ N+1 nodes for N elements, start with node x_0, 1 new node per element """
            self.nodes = np.linspace(self.a, self.b, N + 1)
            self.edge_nodes = [self.nodes[0], self.nodes[-1]]
            self.elements = [[self.nodes[i-1], self.nodes[i]] for i in range(1, N+1)]

            self.h = self.nodes[1:] - self.nodes[:-1]

            self.N = N

        elif p == 2:

            """ 2N+1 nodes for N elements, start with node x_0, 2 new nodes per element """
            self.nodes = np.linspace(self.a, self.b, 2*N + 1)
            self.edge_nodes = [self.nodes[0], self.nodes[-1]]
            self.elements = [[self.nodes[2*i-2], self.nodes[2*i-1], self.nodes[2*i]] for i in range(1, N+1)]

            self.h = self.nodes[2::2] - self.nodes[:-2:2]

            self.N = N

        return


    def build_stiffness_matrix(self, p):

        if self.elements == None:
            raise Exception("Require triangulation before building stiffness matrix.")

        if self.p != None:
            assert p == self.p
        
        self.p = p

        if p == 1:
            Ke = self.Ke_1
            Le = self.Le_1

            self.A = np.zeros((self.N+1,self.N+1), dtype=float)

            """ For each element in triangulation """
            for i in range(self.N):
                self.A[i:i+2,i:i+2] += ( Ke + self.sigma * Le ) / self.h[i]
            
            """ Slice down from proto-problem """
            self.A = self.A[1:-1,1:-1]

        elif p == 2:
            Ke = self.Ke_2
            Le = self.Le_2

            self.A = np.zeros((2*self.N+1,2*self.N+1), dtype=float)

            """ For each element in triangulation """
            for i in range(self.N):
                self.A[2*i : 2*i+3, 2*i : 2*i+3] += ( Ke + self.sigma * Le ) / self.h[i]

            """ Slice down from proto-problem """
            self.A = self.A[1:-1,1:-1]
        
        return


    def build_load_vector(self, p):
        
        if self.elements == None:
            raise Exception("Require triangulation before building stiffness matrix.")

        if self.p != None:
            assert p == self.p

        if p == 1:

            self.F = np.zeros(self.N+1)

            """ For each element in triangulation """
            for i in range(self.N):
                x0, x1 = self.elements[i]
                hi = self.h[i]
                g = lambda x: self.f(x0 + hi * x)
                
                Me1 = gauss_quad(lambda x: (1 - x) * g(x), 0, 1, self.quad_samples)
                Me2 = gauss_quad(lambda x: x * g(x), 0, 1, self.quad_samples)
                Me = np.array([Me1, Me2], dtype=float)

                self.F[i:i+2] += hi * Me

            """ Apply boundary conditions """
            self.F[1] -= -self.u_1 / self.h[0] + self.u_1 * self.sigma * self.h[0] / 6
            self.F[self.N-1] -= -self.u_2 / self.h[self.N-1] + self.u_2 * self.sigma * self.h[self.N-1] / 6

            """ Slice down from proto-problem """
            self.F = self.F[1:-1]

        elif p == 2:

            self.F = np.zeros(2*self.N+1)

            """ For each element in triangulation """
            for i in range(self.N):
                x0, x1, x2 = self.elements[i]
                hi = self.h[i]
                g = lambda x: self.f(x0 + hi * x)
                
                Me1 = gauss_quad(lambda x: (x - 1) * (2*x - 1)  * g(x), 0, 1, self.quad_samples)
                Me2 = gauss_quad(lambda x: 4*(1 - x) * x * g(x), 0, 1, self.quad_samples)
                Me3 = gauss_quad(lambda x: x * (2*x - 1) * g(x), 0, 1, self.quad_samples)
                Me = np.array([Me1, Me2, Me3], dtype=float)

                self.F[2*i:2*i+3] += hi * Me

            """ Apply boundary conditions """
            self.F[1] -= -self.u_1 / self.h[0] * 8/3 + self.u_1 * self.sigma * self.h[0] / 15
            self.F[2] -=  self.u_1 / self.h[0] * 1/3 - self.u_1 * self.sigma * self.h[0] / 30

            self.F[2*self.N-2] -=  self.u_2 / self.h[self.N-1] * 1/3 - self.u_2 * self.sigma * self.h[self.N-1] / 30
            self.F[2*self.N-1] -= -self.u_2 / self.h[self.N-1] * 8/3 + self.u_2 * self.sigma * self.h[self.N-1] / 15

            """ Slice down from proto-problem """
            self.F = self.F[1:-1]


        return


    def solve_linear_system(self):
        
        self.u = np.zeros_like(self.nodes)

        self.u[1:-1] = np.linalg.solve(self.A, self.F)

        self.u[0] = self.u_1
        self.u[-1] = self.u_2

        return

    def solve_linear_system_sparse(self):

        self.u = np.zeros_like(self.nodes)
        A_s = scipy.sparse.csc_matrix(self.A)

        u_h, exit_code = scipy.sparse.linalg.cg(A_s, self.F)

        if exit_code == 0:
            self.u[1:-1] = u_h

            self.u[0] = self.u_1
            self.u[-1] = self.u_2

        else:
            raise Exception("CG did not converge")

        return

    def error(self, u_ex):
        
        assert type(self.u) != None

        if self.p == 1:

            raise NotImplementedError

        
        """ This does not work """

        elif self.p == 2:
            
            E = 0

            for i in range(self.N):
                
                x0, x1, x2 = self.elements[i]
                u0, u1, u2 = self.u[2*i:2*i+3]
                
                chi = lambda x: x0 + (x2 - x0) * x
                chi = lambda x: x / self.h[i] + x0

                phi0 = lambda x: (x - 1) * (2*x - 1)
                phi1 = lambda x: 4*(1 - x) * x
                phi2 = lambda x: x * (2*x - 1)

                uh = lambda x: u0 * phi0(chi(x)) + u1 * phi1(chi(x)) + u2 * phi2(chi(x))

                E +=  self.h[i]*gauss_quad(lambda x: ( u_ex(x) - uh(x) )**2, x0, x2, 5)

            return np.sqrt(E)


    def plot_solution(self):
        
        xx = self.nodes
        yy = self.u

        plt.plot(xx, yy, 'k--', label="$u_h$")

        return


def main():
    from time import time

    sigma = 0
    u_ex = lambda x: np.sin(np.pi * x)
    f = lambda x: ( np.pi**2 + sigma ) * np.sin(np.pi * x)
    a = 0
    b = 2
    u_1 = 0
    u_2 = 0

    femsolver = Femsolver(sigma, f, a, b, u_1, u_2)

    N = 1000
    p = 2

    femsolver.build_triangulation(N, p)

    start = time()
    femsolver.build_stiffness_matrix(p)
    end = time()
    print(f'Dense building: {(end-start)*1e3:.2f} ms')   

    femsolver.build_load_vector(p)
    
    start = time()
    femsolver.solve_linear_system()
    end = time()
    print(f'Normal solving: {(end-start)*1e3:.2f} ms')
    start = time()
    #femsolver.solve_linear_system_sparse()
    end = time()
    #print(f'Sparse solving: {(end-start)*1e3:.2f} ms')

    print(f'Error = {femsolver.error(u_ex)}')

    femsolver.plot_solution()

    """ plot exact solution """
    xx = np.linspace(a, b, 200)
    yy = np.sin(np.pi * xx)
    plt.plot(xx, yy, 'k-', label=r"$u_{ex}$")

    """ plot error """
    xx = femsolver.nodes
    yy = np.sin(np.pi * xx) - femsolver.u
    plt.plot(xx, yy, 'k:', label=r"$u_{ex} - u_h$")

    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    main()
    