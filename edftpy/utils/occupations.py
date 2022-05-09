import numpy as np
from scipy.special import erf
from functools import partial

from edftpy.mpi import sprint

def gaussian(x, **kwargs):
    return methfessel_paxton(x, order=0)

def fermi_dirac(x, cut = 100.0, **kwargs):
    x = np.clip(x, -cut, cut)
    ex = np.exp(x)
    s = 1.0 /(1.0+ex)
    return s

def marzari_vanderbilt(x, **kwargs):
    """
    also name as cold_smearing
    ref :
       https://doi.org/10.1103/physrevlett.82.3296
    """
    xm = x + 1.0 / np.sqrt(2.0)
    ex2 = np.exp(-(xm*xm))
    s = ex2 / np.sqrt(2 * np.pi) + 0.5 * (1 - erf(xm))
    return s

def methfessel_paxton(x, order = 1, **kwargs):
    """
    ref :
        https://doi.org/10.1103/physrevb.40.3616
    """
    s0 = 0.5 * (1 - erf(x))
    if order < 1 : return s0
    a = 1.0 / np.sqrt(np.pi)
    k = 0
    ex2 = np.exp(-(x*x))
    h1 = np.ones_like(x)
    h2 = 2.0*x
    s = np.zeros_like(s0)
    for i in range(1, order + 1):
        a /= (-4*i)
        s += a*h2
        k = k + 1
        h2, h1 = 2.0 * x * h2 - 2 * k * h1, h2
        k = k + 1
        h2, h1 = 2.0 * x * h2 - 2 * k * h1, h2
    s = s0 + s*ex2
    return s

def insulator(x, **kwargs):
    s = np.where(x > 0.0, 0.0, 1.0)
    return s


OCCEngines = {
    'gaussian' : gaussian,
    'fermi_dirac' : fermi_dirac,
    'methfessel_paxton' : methfessel_paxton,
    'marzari_vanderbilt' : marzari_vanderbilt,
    'insulator' : insulator,
}

class Occupations:
    """description"""
    def __init__(self, smearing = 'gaussian', degauss = 0.0, **kwargs):
        self.smearing = smearing
        self.degauss = degauss
        if self.smearing not in OCCEngines :
            raise AttributeError(f"Sorry the {self.smearing} occupations method is not supported.")

    @property
    def degauss(self):
        return self._degauss

    @degauss.setter
    def degauss(self, value):
        if value > 0.0 :
            if self.smearing == 'insulator' : self.smearing = 'gaussian'
            self._degauss = value
        else :
            self.smearing = 'insulator'
            self._degauss = 1.0

    def get_occupation_numbers(self, elevels, ef = None, **kwargs):
        """
        Note :
            For speed, only sort the elevels when 'ef' not given.
        """
        if ef is None :
            inds = np.argsort(elevels)
            elevels = elevels[inds]
            ef = self.get_fermi_level(elevels, **kwargs)
        else :
            inds = None
        x=(elevels-ef)/self.degauss
        occs = OCCEngines[self.smearing](x, **kwargs)
        if inds is not None :
            occs[inds] = occs.copy()
        return occs

    def get_fermi_level(self, elevels, ncharge = None, weights = 1.0, emin = None, emax = None, maxiter = 500, tol = 1E-10, **kwargs):
        """
        ref :
           https://doi.org/10.1016/j.cpc.2019.06.017
        """
        if ncharge is None :
            raise AttributeError("Please give the number of electrons : 'ncharge'")

        weights = np.asarray(weights)
        if weights.size == 1 : weights = np.ones_like(elevels)*weights

        if self.smearing == 'insulator' :
            nc = 0.0
            for iband in range(len(elevels)):
                nc += weights[iband]
                if abs(nc - ncharge) < tol :
                    break
            if abs(nc - ncharge) > tol :
                raise AttributeError("Can not find the good fermi level.")
            ef = elevels[iband] + 1E-30 # for safe
            return ef

        if emin is None : emin = elevels[0] - 2*self.degauss
        if emax is None : emax = elevels[-1] + 2*self.degauss

        def func(ef):
            occs = self.get_occupation_numbers(elevels, ef=ef, **kwargs)
            diff = (occs*weights).sum() - ncharge
            return diff

        ef, diff = self.get_opt_number(func, emin, emax, maxiter = maxiter, tol = tol, **kwargs)
        if abs(diff) > tol :
            raise AttributeError(f"Can not find the good fermi level. ({ef} -> {diff})")

        return ef

    @staticmethod
    def get_opt_number(func, emin, emax, maxiter = 500, tol = 1E-10, **kwargs):
        ef = (emin + emax)/2.0
        step = min(1.0, (emax - ef)*(0.5+tol))

        lfirst = True
        for i in range(maxiter):
            diff = func(ef)
            sprint('Ef opt -> ', i, ef, diff, level = 1)
            if diff * step > 0.0 :
                step *= -0.25
            if abs(diff) < tol:
                break
            if ef > emax or ef < emin :
                if lfirst :
                    lfirst = False
                    if ef > emax : ef = emax
                    if ef < emin : ef = emin
                else :
                    break
            ef += step
        return ef, diff

    def get_occupation_numbers_v(self, elevels, ef = None, **kwargs):
        if ef is None :
            ef = self.get_fermi_level_v(elevels, **kwargs)
        occs = []
        for es, e in zip(elevels, ef):
            x = self.get_occupation_numbers(es, ef = e, **kwargs)
            occs.append(x)
        return occs

    def get_fermi_level_v(self, elevels, ncharge = None, weights = 1.0, emin = None, emax = None, maxiter = 500, tol = 1E-10, diffs = 0.0, eratio = 0.1, **kwargs):
        if ncharge is None :
            raise AttributeError("Please give the number of electrons : 'ncharge'")

        def diff_nele(ef, efs = None, diffs = 0.0, i = 0):
            if efs is None :
                efs = ef + diffs
            else :
                efs[i] = ef
            occs = self.get_occupation_numbers_v(elevels, ef=efs, **kwargs)
            diff = sum([(x*y).sum() for x, y in zip(occs, weights)]) - ncharge
            return diff

        if emin is None : emin = elevels[0][0] - 2*self.degauss
        if emax is None : emax = elevels[0][-1] + 2*self.degauss

        diffs = np.concatenate((np.zeros(1), np.atleast_1d(diffs)))

        func = partial(diff_nele, diffs = diffs)
        ef, diff = self.get_opt_number(func, emin, emax, maxiter = maxiter, tol = tol, **kwargs)
        efs = ef + diffs
        sprint('ef', 0, ef, efs, diff, level = 1)
        if abs(diff) > tol :
            for i, e in enumerate(efs):
                j = 1 if i == 0 else i
                emin0 = efs[i]- eratio*abs(efs[j])
                emax0 = efs[i]+ eratio*abs(efs[j])
                func = partial(diff_nele, efs = efs, i = i)
                ef, diff = self.get_opt_number(func, emin0, emax0, maxiter = maxiter, tol = tol, **kwargs)
                sprint('ef', i, ef, efs, diff, flush = True, level = 1)
                if abs(diff) < tol : break

        if abs(diff) > tol :
            raise AttributeError(f"Can not find the good fermi level. ({efs[0]} -> {diff})")

        return efs

    def get_fermi_level_pot(self, elevels, ncharge = None, weights = 1.0, emin = None, emax = None, maxiter = 500, tol = 1E-10, diffs = 0.0, eratio = 0.1, index = (0, 1), **kwargs):
        if ncharge is None :
            raise AttributeError("Please give the number of electrons : 'ncharge'")

        inds = [len(x) for x in elevels]
        inds.insert(0, 0)
        inds = np.cumsum(inds)

        elevels_total = np.concatenate(elevels)
        weights_total = np.concatenate(weights)
        sind = np.argsort(elevels_total)
        elevels_total = elevels_total[sind]
        weights_total = weights_total[sind]
        sind = np.argsort(sind)
        ind0 = sind[inds[index[0]]:inds[index[0]+1]]
        ind1 = sind[inds[index[1]]:inds[index[1]+1]]

        def func(ef):
            # print('eeee', elevels_total, ef)
            occs = self.get_occupation_numbers(elevels_total, ef=ef, **kwargs)
            diff = (occs*weights_total).sum() - ncharge
            return diff

        if emin is None : emin = elevels_total[0] - 2*self.degauss
        if emax is None : emax = elevels_total[-1] + 2*self.degauss

        dv = 4*self.degauss*np.sign(diffs)
        nv = int(abs(diffs//dv)) + 1
        ef_diff = -100
        for it in range(nv):
            ef, diff = self.get_opt_number(func, emin, emax, maxiter = maxiter, tol = tol, **kwargs)
            sprint('ef -> ', it, ef, diff, flush = True, level = 1)
            if abs(diff) < tol :
                occs = self.get_occupation_numbers(elevels_total, ef=ef, **kwargs)
                ef_0 = self.guess_ef(elevels_total[ind0], occs[ind0])
                ef_1 = self.guess_ef(elevels_total[ind1], occs[ind1])
                ef_diff = ef_0 - ef_1
                sprint('ef_try -> ', ef, ef_0, ef_1, ef_diff, level = 1)
                if abs((ef_diff - diffs)/diffs) < eratio : break
            elevels[index[0]] += dv
            elevels[index[1]] -= dv
            elevels_total = np.concatenate(elevels)
            weights_total = np.concatenate(weights)
            sind = np.argsort(elevels_total)
            elevels_total = elevels_total[sind]
            weights_total = weights_total[sind]
            sind = np.argsort(sind)
            ind0 = sind[inds[index[0]]:inds[index[0]+1]]
            ind1 = sind[inds[index[1]]:inds[index[1]+1]]

        if abs(diff) > tol :
            raise AttributeError(f"Can not find the good fermi level. ({ef} -> {diff})")
        elif abs((ef_diff - diffs)/diffs) > eratio :
            raise AttributeError(f"Can not find the correct one. ({ef} -> {ef_diff})")

        return ef

    def guess_ef(self, elevels, occs, degauss = None):
        if degauss is None : degauss = self.degauss
        oc = np.abs(occs - 0.5)
        i = np.argmin(oc)
        if oc[i] < 2*degauss :
            ef = elevels[i]
        else :
            e_up = elevels[0]
            e_dw = elevels[-1]
            for x, y in zip(elevels, occs):
                if y > 1 - degauss :
                    e_up = x
                elif y < degauss :
                    e_dw = x
                    break
            ef = (e_up + e_dw)/2
        return ef
