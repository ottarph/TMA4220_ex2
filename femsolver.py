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

            E = 0

            for i in range(self.N):
                
                x0, x1 = self.elements[i]
                u0, u1 = self.u[i:i+2]
                
                chi = lambda x: (x - x0) / self.h[i]

                phi0 = lambda x: 1 - x
                phi1 = lambda x: x

                uh = lambda x: u0 * phi0(chi(x)) + u1 * phi1(chi(x))

                E += gauss_quad(lambda x: ( u_ex(x) - uh(x) )**2, x0, x1, 5)
            
            return np.sqrt(E)

        
        elif self.p == 2:
            
            E = 0

            for i in range(self.N):
                
                x0, x1, x2 = self.elements[i]
                u0, u1, u2 = self.u[2*i:2*i+3]
                
                chi = lambda x: (x - x0) / self.h[i]

                phi0 = lambda x: (x - 1) * (2*x - 1)
                phi1 = lambda x: 4*(1 - x) * x
                phi2 = lambda x: x * (2*x - 1)

                uh = lambda x: u0 * phi0(chi(x)) + u1 * phi1(chi(x)) + u2 * phi2(chi(x))

                E += gauss_quad(lambda x: ( u_ex(x) - uh(x) )**2, x0, x2, 5)

            return np.sqrt(E)


    def plot_solution(self, simple=True, ax=None):
        

        if simple:
            xx = self.nodes
            yy = self.u

            if ax is None:
                plt.plot(xx, yy, 'k--', label="$u_h$")
            else:
                ax.plot(xx, yy, 'k--', label="$u_h$")

            return

        else:

            if ax is None:
                ax = plt.gca()

            points = 1000
            elpoints = points // len(self.elements)

            for i in range(self.N):

                if self.p == 1:

                    x0, x1 = self.elements[i]
                    u0, u1 = self.u[i:i+2]
                    
                    chi = lambda x: (x - x0) / self.h[i]

                    phi0 = lambda x: 1 - x
                    phi1 = lambda x: x

                    uh = lambda x: u0 * phi0(chi(x)) + u1 * phi1(chi(x))

                    xx = np.linspace(x0, x1, elpoints)

                    ax.plot(xx, uh(xx), 'k-')

                elif self.p == 2:
                
                    x0, x1, x2 = self.elements[i]
                    u0, u1, u2 = self.u[2*i:2*i+3]
                    
                    chi = lambda x: (x - x0) / self.h[i]

                    phi0 = lambda x: (x - 1) * (2*x - 1)
                    phi1 = lambda x: 4*(1 - x) * x
                    phi2 = lambda x: x * (2*x - 1)

                    uh = lambda x: u0 * phi0(chi(x)) + u1 * phi1(chi(x)) + u2 * phi2(chi(x))

                    xx = np.linspace(x0, x2, elpoints)

                    ax.plot(xx, uh(xx), 'k-')

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
    p = 1

    E1 = []
    E2 = []

    Ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    for N in Ns:

        print(f'\n\nN = {N}')
        if N < 32:
            fig, axs = plt.subplots(1,2)

        p = 1
        print(f'p = {p}', end='\t')

        femsolver.build_triangulation(N, p)
        femsolver.build_stiffness_matrix(p)
        femsolver.build_load_vector(p)
        
        femsolver.solve_linear_system()

        E = femsolver.error(u_ex)
        E1.append(E)

        if N < 64:
            femsolver.plot_solution(ax=axs[0], simple=False)

        p = 2
        print(f'p = {p}', end='\t')

        femsolver.build_triangulation(N, p)
        femsolver.build_stiffness_matrix(p)
        femsolver.build_load_vector(p)
        
        femsolver.solve_linear_system()

        E = femsolver.error(u_ex)
        E2.append(E)

        if N < 32:
            femsolver.plot_solution(ax=axs[1], simple=False)

    Ns = np.array(Ns)
    E1 = np.array(E1)
    E2 = np.array(E2)

    plt.figure()
    plt.loglog(Ns, E1, 'k--')
    plt.loglog(Ns, E2, 'k:')

    def beta(x, y):
        '''
            Estimator for the coefficient of beta in linear regression model
                y = alpha + beta * x
        '''
        n = x.shape[0]
        
        beta = np.sum( (x - np.mean(x)) * (y - np.mean(y))) / np.sum( (x - np.mean(x))**2 )

        return beta


    beta1 = beta(np.log(Ns), np.log(E1))
    print(f'{-beta1:.2f}')
    beta2 = beta(np.log(Ns), np.log(E2))
    print(f'{-beta2:.2f}')

    plt.show()

    return


if __name__ == '__main__':
    main()
    