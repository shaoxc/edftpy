import numpy as np
import scipy.special as sp
from scipy import linalg
from abc import ABC, abstractmethod

from .utils.common import Field

__all__ = ["LinearMixer", "PulayMixer"]

class SpecialPrecondition :
    def __init__(self, predtype = 'kerker', predcoef = [0.8, 1.0], grid = None):
        self.predtype = predtype
        self.predcoef = predcoef
        self._grid = grid
        self._matrix = None
        self._direct = False

    @property
    def matrix(self):
        if self._matrix is None :
            if self.predtype is None :
                self._matrix = 1.0
            elif self.predtype == 'kerker' :
                self._matrix = self.kerker()
            elif self.predtype == 'inverse_kerker' :
                self._matrix = self.inverse_kerker()
        return self._matrix

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value
        self._matrix = None

    def kerker(self):
        a0 = self.predcoef[0]
        q0 = self.predcoef[1] ** 2
        recip_grid = self.grid.get_reciprocal()
        gg = recip_grid.gg
        preg = a0 * gg/(gg+q0)
        preg = Field(recip_grid, data=preg, direct=False)
        matrix = preg
        # matrix = preg.ifft(force_real=True)
        return matrix

    def inverse_kerker(self):
        b0 = self.predcoef[0] ** 2
        recip_grid = self.grid.get_reciprocal()
        gg = recip_grid.gg
        gg[0, 0, 0] = 1.0
        preg = b0/gg + 1.0
        gg[0, 0, 0] = 0.0
        preg[0, 0, 0] = 0.0
        preg = Field(recip_grid, data=preg, direct=False)
        matrix = preg
        # matrix = preg.ifft(force_real=True)
        return matrix

    def __call__(self, density, residual = None, grid = None):
        results = self.compute(density, residual, grid)
        return results

    def compute(self, density, residual = None, grid = None):
        if grid is not None :
            self.grid = grid
        if not self._direct :
            den = Field(self.grid, data=density, direct=True)
            if residual is None :
                results = (self.matrix*den.fft()).ifft(force_real=True)
            else :
                res = Field(self.grid, data=residual, direct=True)
                results = (den.fft() + self.matrix*res.fft()).ifft(force_real=True)
        else :
            raise AttributeError("Real-space matrix will implemented soon")
        return results


class AbstractMixer(ABC):
    """
    This is a template class for mixer
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    def format_density(self, results, nin, tol = 1E-30):
        results[results < tol] = tol
        ncharge = np.sum(nin)
        results *= ncharge/np.sum(results)
        results = Field(nin.grid, data=results, direct=True)
        return results


class LinearMixer(AbstractMixer):
    def __init__(self, predtype = None, predcoef = [0.8, 1.0], coef = [0.5], delay = 3, **kwargs):
        # if predtype is None :
            # self.pred = None
        self.pred = SpecialPrecondition(predtype, predcoef)
        self.coef = coef
        self._delay = delay
        self._iter = 0

    def __call__(self, nin, nout, coef = None):
        results = self.compute(nin, nout, coef)
        return results

    def compute(self, nin, nout, coef = None):
        self._iter += 1
        if coef is None :
            coef = self.coef
        if self._iter > self._delay :
            res = self.residual(nin, nout)
            res *= coef[0]
            results = self.pred(nin, res, nin.grid)
        else :
            results = nout.copy()

        results = self.format_density(results, nin)
        return results

    def residual(self, nin, nout):
        res = nout - nin
        return res


class PulayMixer(AbstractMixer):
    def __init__(self, predtype = None, predcoef = [0.8, 0.1], maxm = 5, coef = [1.0], delay = 3):
        self.pred = SpecialPrecondition(predtype, predcoef)
        self._iter = 0
        self._delay = delay
        self.maxm = maxm
        self.dr_mat = None
        self.dn_mat = None
        self.coef = coef
        self.prev_density = None
        self.prev_residual = None

    def __call__(self, nin, nout, coef = None):
        results = self.compute(nin, nout, coef)
        return results

    def residual(self, nin, nout):
        res = nout - nin
        return res

    def compute(self, nin, nout, coef = None):
        """
        Ref : G. Kresse and J. Furthmuller, Comput. Mat. Sci. 6, 15-50 (1996).
        """
        if coef is None :
            coef = self.coef
        self._iter += 1

        r = nout - nin
        if self._iter > self._delay :
            dn = nin - self.prev_density
            dr = r - self.prev_residual
            if self.dr_mat is None :
                self.dr_mat = dr.reshape((-1, *dr.shape))
                self.dn_mat = dn.reshape((-1, *dn.shape))
            elif len(self.dr_mat) < self.maxm :
                self.dr_mat = np.concatenate((self.dr_mat, dr.reshape((-1, *r.shape))))
                self.dn_mat = np.concatenate((self.dn_mat, dn.reshape((-1, *r.shape))))
            else :
                self.dr_mat = np.roll(self.dr_mat,-1,axis=0)
                self.dr_mat[-1] = dr
                self.dn_mat = np.roll(self.dn_mat,-1,axis=0)
                self.dn_mat[-1] = dn

            ns = len(self.dr_mat)
            amat = np.empty((ns, ns))
            b = np.empty((ns))
            for i in range(ns):
                for j in range(i + 1):
                    amat[i, j] = np.sum(self.dr_mat[i] * self.dr_mat[j])
                    amat[j, i] = amat[i, j]
                b[i] = -np.sum(self.dr_mat[i] * r)

            x = linalg.solve(amat, b, assume_a = 'sym')
            print('x', x)

            for i in range(ns):
                if i == 0 :
                    results = nin + x[i] * self.dn_mat[i]
                    res = r + x[i] * self.dr_mat[i]
                else :
                    results += x[i] * self.dn_mat[i]
                    res += x[i] * self.dr_mat[i]
            res *= coef[0]
            results = self.pred(results, res, nin.grid)
        else :
            results = nout.copy()
        #-----------------------------------------------------------------------
        results = self.format_density(results, nin)
        #-----------------------------------------------------------------------
        self.prev_density = nin.copy()
        self.prev_residual = r.copy()

        return results
