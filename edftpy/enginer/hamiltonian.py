import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, eigs
from edftpy.utils.common import Field, Grid

class Hamiltonian(object):

    # def __init__(self, potential=None, sigma = None, laplacian = False, grid = None):
    def __init__(self, potential=None, sigma = None, laplacian = True, grid = None):
        self.sigma = sigma
        self.laplacian = laplacian
        self._potential = potential
        self._grid = grid

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, value):
        self._potential = value

    @property
    def grid(self):
        if self._grid is None :
            self._grid = self.potential.grid
        if self._grid is None :
            raise AttributeError("Must given grid firstly.")
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    def h_mul_phi(self, phi):
        if self.laplacian :
            results = self.potential*phi - 0.5 * phi.laplacian(force_real = True, sigma=self.sigma)
        else :
            results = self.potential*phi
        return results

    def matvec(self, value):
        phi = Field(self.grid, data = value)
        results = self.h_mul_phi(phi)
        return results.ravel()

    def eigens(self, num_eig = 1, which = 'SA', return_eigenvectors = True, eig_tol = 1E-10, **kwargs):
        dtype = np.float64
        size = self.grid.nnr
        amat = LinearOperator((size, size), dtype=dtype, matvec=self.matvec)
        eigens = eigsh(amat, k=num_eig, which=which, return_eigenvectors=return_eigenvectors, tol = eig_tol)
        if not return_eigenvectors :
            results = eigens
        else :
            results = []
            for i, eig in enumerate(eigens[0]):
                phi = Field(self.grid, data = eigens[1][:, i])
                results.append([eig, phi])
                res = self.h_mul_phi(phi) - eig * phi
                print('res', np.max(res), np.min(res), np.sum(res * res)*self.grid.dV)
                # print('phi', np.max(phi), np.min(phi), np.sum(phi * phi))
        return results
