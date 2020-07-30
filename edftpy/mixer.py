import numpy as np
import scipy.special as sp
from scipy import linalg
from abc import ABC, abstractmethod

from .utils.common import Field
from .utils.math import fermi_dirac
from .density import normalization_density

__all__ = ["LinearMixer", "PulayMixer", "BroydenMixer"]

class SpecialPrecondition :
    def __init__(self, predtype = 'inverse_kerker', predcoef = [0.8, 1.0, 1.0], grid = None, predecut = None, **kwargs):
        self.predtype = predtype
        self._init_predcoef(predcoef, predtype)
        self._ecut = predecut
        self._grid = grid
        self._matrix = None
        self._direct = False
        self._mask = False

    def _init_predcoef(self, predcoef=[], predtype = 'inverse_kerker'):
        nl = len(predcoef)
        if nl == 0 :
            predcoef = [0.8, 1.0, 1.0]
        elif nl == 1 :
            predcoef.extend([1.0, 1.0])
        elif nl == 2 :
            predcoef.extend([1.0])
        self.predcoef = predcoef

    @property
    def matrix(self):
        if self._matrix is None :
            if self.predtype is None :
                self._matrix = np.ones(self.grid.nrG)
            elif self.predtype == 'kerker' :
                self._matrix = self.kerker()
            elif self.predtype == 'inverse_kerker' :
                self._matrix = self.inverse_kerker()
            elif self.predtype == 'resta' :
                self._matrix = self.resta()
        return self._matrix

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value
        self._matrix = None
        self._mask = None

    @property
    def mask(self):
        if self._mask is None :
            gg = self.grid.get_reciprocal().gg
            self._mask = np.zeros(self.grid.nrG, dtype = 'bool')
            if self._ecut is not None :
                if self._ecut < 2 :
                    gmax = max(gg[:, 0, 0].max(), gg[0, :, 0].max(), gg[0, 0, :].max()) + 2
                else :
                    gmax = 2.0 * self._ecut
                self._mask[gg > gmax] = True
                print('Density mixing gmax', gmax, self._ecut)
        return self._mask

    def kerker(self):
        a0 = self.predcoef[0]
        q0 = self.predcoef[1] ** 2
        amin = self.predcoef[2]
        recip_grid = self.grid.get_reciprocal()
        gg = recip_grid.gg
        preg = a0 * np.minimum(gg/(gg+q0), amin)
        # preg = a0 * np.maximum(gg/(gg+q0), 0.01)
        preg = Field(recip_grid, data=preg, direct=False)
        matrix = preg
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
        return matrix

    def resta(self):
        epsi = self.predcoef[0]
        q0 = self.predcoef[1] ** 2
        rs = self.predcoef[2]
        recip_grid = self.grid.get_reciprocal()
        gg = recip_grid.gg
        q = recip_grid.q
        q[0, 0, 0] = 1.0
        preg = (q0 * np.sin(q*rs)/(epsi*q*rs)+gg) / (q0+gg)
        q[0, 0, 0] = 0.0
        preg[0, 0, 0] = 1.0
        preg = Field(recip_grid, data=preg, direct=False)
        matrix = preg
        return matrix

    def __call__(self, nin, nout, drho = None, residual = None):
        results = self.compute(nin, nout, drho, residual)
        return results

    def compute(self, nin, nout, drho = None, residual = None):
        if self.grid is None :
            self.grid = nin.grid
        nin_g = nin.fft()
        results = nin_g.copy()
        if drho is not None :
            dr = Field(self.grid, data=drho, direct=True)
            results += dr.fft()
        if residual is not None :
            res = Field(self.grid, data=residual, direct=True)
            results += res.fft()*self.matrix
        results[self.mask] = nout.fft()[self.mask]
        # results[self.mask] = nin_g[self.mask]*0.8 + 0.2 * nout.fft()[self.mask]
        # results[self.mask] = nin_g[self.mask]
        return results.ifft(force_real=True)

    def add(self, density, residual = None, grid = None):
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

    def format_density(self, results, nin):
        ncharge = nin.integral()
        results = normalization_density(results, ncharge=ncharge, grid=nin.grid)
        return results


class LinearMixer(AbstractMixer):
    def __init__(self, predtype = None, predcoef = [0.8, 1.0], coef = [0.5], delay = 3, predecut = None, **kwargs):
        # if predtype is None :
            # self.pred = None
        self.pred = SpecialPrecondition(predtype, predcoef, predecut=predecut)
        self.coef = coef
        self._delay = delay
        self.restart()

    def restart(self):
        self._iter = 0

    def __call__(self, nin, nout, coef = None):
        results = self.compute(nin, nout, coef)
        return results

    def compute(self, nin, nout, coef = None):
        self._iter += 1
        one = 1.0-1E-10
        if coef is None :
            coef = self.coef
        if self._iter > self._delay and coef[0] < one:
            res = nout - nin
            res *= coef[0]
            results = self.pred(nin, nout, residual=res)
            results = self.format_density(results, nin)
        else :
            results = nout.copy()

        return results

    def residual(self, nin, nout):
        res = nout - nin
        return res


class PulayMixer(AbstractMixer):
    def __init__(self, predtype = None, predcoef = [0.8, 0.1], maxm = 5, coef = [1.0], delay = 3, predecut = None, restarted = False, **kwargs):
        self.pred = SpecialPrecondition(predtype, predcoef, predecut=predecut)
        self._delay = delay
        self.maxm = maxm
        self.coef = coef
        self.restarted = restarted
        self.restart()

    def __call__(self, nin, nout, coef = None):
        results = self.compute(nin, nout, coef)
        return results

    def restart(self):
        self._iter = 0
        self.dr_mat = None
        self.dn_mat = None
        self.prev_density = None
        self.prev_residual = None

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

        print('ccccc', coef)
        # rho0 = np.mean(nin)
        # kf = (3.0 * rho0 * np.pi ** 2) ** (1.0 / 3.0)
        # print('kf', kf, self.pred.predcoef[1])
        # if self._iter == 20 :
        # if abs(kf - self.pred.predcoef[1]) > 0.1 :
            # self.pred.predcoef[1] = kf
            # self.pred._matrix = None
            # print('Embed restart pulay coef')
            # self.dr_mat = None
            # self.dn_mat = None
            # print('Restart history of the mixer')

        r = nout - nin
        #-----------------------------------------------------------------------
        if self._iter == 1 and self._delay < 1 :
            res = r * coef[0]
            print('!WARN : Change to linear mixer')
            results = self.pred(nin, nout, residual=res)
        elif self._iter > self._delay :
            if self.restarted and (self._iter-self._delay) %(self.maxm+1)==0 :
                self.dr_mat = None
                self.dn_mat = None
                print('Restart history of the mixer')
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

            try:
                x = linalg.solve(amat, b, assume_a = 'sym')
                # print('x', x)
                for i in range(ns):
                    if i == 0 :
                        drho = x[i] * self.dn_mat[i]
                        res = r + x[i] * self.dr_mat[i]
                    else :
                        drho += x[i] * self.dn_mat[i]
                        res += x[i] * self.dr_mat[i]
                # drho *= coef[0]
                #-----------------------------------------------------------------------
                # res[:] = filter_density(res)
                #-----------------------------------------------------------------------
                res *= coef[0]
                results = self.pred(nin, nout, drho, res)
            except Exception :
                res = r * coef[0]
                print('!WARN : Change to linear mixer')
                print('amat', amat)
                results = self.pred(nin, nout, residual=res)
        else :
            # res = r * 0.8
            # results = self.pred(nin, nout, residual=res)
            # print('delay : use linear mixer')
            results = nout.copy()
            print('delay : use output density')
        #-----------------------------------------------------------------------
        results = self.format_density(results, nin)
        #-----------------------------------------------------------------------
        self.prev_density = nin.copy()
        self.prev_residual = r.copy()

        return results

    def get_direction(self, r):
        ns = len(self.dr_mat)
        amat = np.empty((ns, ns))
        b = np.empty((ns))
        for i in range(ns):
            for j in range(i + 1):
                amat[i, j] = np.sum(self.dr_mat[i] * self.dr_mat[j])
                amat[j, i] = amat[i, j]
            b[i] = -np.sum(self.dr_mat[i] * r)

        try:
            x = linalg.solve(amat, b, assume_a = 'sym')
            # print('x', x)
            for i in range(ns):
                if i == 0 :
                    drho = x[i] * self.dn_mat[i]
                    res = r + x[i] * self.dr_mat[i]
                else :
                    drho += x[i] * self.dn_mat[i]
                    res += x[i] * self.dr_mat[i]
            return (res, drho)
        except Exception :
            print('!WARN : Change to linear mixer')
            print('amat', amat)
            return False


class BroydenMixer(AbstractMixer):
    """
    Not finished !!! 
    """
    def __init__(self, predtype = None, predcoef = [0.8, 0.1], maxm = 5, coef = [1.0], delay = 3):
        self.pred = SpecialPrecondition(predtype, predcoef)
        self._delay = delay
        self.maxm = maxm
        self.coef = coef
        self.restart()

    def __call__(self, nin, nout, coef = None):
        results = self.compute(nin, nout, coef)
        return results

    def restart(self):
        self._iter = 0
        self.dr_mat = None
        self.dn_mat = None
        self.jr_mat = None
        self.prev_density = None
        self.prev_residual = None

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
                self.jr_mat = np.empty_like(self.dr_mat)
            elif len(self.dr_mat) < self.maxm :
                self.dr_mat = np.concatenate((self.dr_mat, dr.reshape((-1, *r.shape))))
                self.dn_mat = np.concatenate((self.dn_mat, dn.reshape((-1, *r.shape))))
                self.jr_mat = np.concatenate((self.jr_mat, np.empty((1, *r.shape))))
            else :
                self.dr_mat = np.roll(self.dr_mat,-1,axis=0)
                self.dr_mat[-1] = dr
                self.dn_mat = np.roll(self.dn_mat,-1,axis=0)
                self.dn_mat[-1] = dn
                self.jr_mat = np.roll(self.jr_mat,-1,axis=0)

            ns = len(self.dr_mat)
            amat = np.empty((ns, ns))
            x = np.empty((ns))
            for i in range(ns):
                for j in range(i + 1):
                    amat[i, j] = np.sum(self.dr_mat[i] * self.dr_mat[j])
                    amat[j, i] = amat[i, j]
            self.jr_mat[-1] = self.dn_mat[-1].copy()
            self.jr_mat[-1] = self.pred(self.jr_mat[-1], self.dr_mat[-1]*coef[0], grid=nin.grid)
            for i in range(ns - 1):
                self.jr_mat[-1] -= amat[i, -1] * self.jr_mat[-1]
            self.jr_mat[-1] /= amat[-1, -1]
            for i in range(ns):
                x[i] = -np.sum(self.dr_mat[i] * r)

            for i in range(ns):
                if i == 0 :
                    results = nin + x[i] * self.jr_mat[i]
                else :
                    results += x[i] * self.jr_mat[i]
            results = self.pred(results, r*coef[0], nin.grid)
        else :
            results = nout.copy()
        #-----------------------------------------------------------------------
        results = self.format_density(results, nin)
        #-----------------------------------------------------------------------
        self.prev_density = nin.copy()
        self.prev_residual = r.copy()

        return results

# def filter_density(density, mu = 0.5, kt = 0.01):
    # ncharge = np.sum(density)
    # recip_grid = density.grid.get_reciprocal()
    # q = recip_grid.q
    # mu *= np.max(q)
    # kt *= mu
    # _filter = fermi_dirac(q.real, mu = mu, kt = kt)
    # den_g = density.fft() * _filter
    # den = den_g.ifft(force_real=True)
    # den *= ncharge/np.sum(den)
    # return den
