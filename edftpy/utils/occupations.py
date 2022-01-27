import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
# from edftpy.mpi import sprint

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

    def get_fermi_level(self, elevels, ncharge = None, weights = 1.0, nbands = -1, maxiter = 500, tol = 1E-10, **kwargs):
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

        def func(ef):
            occs = self.get_occupation_numbers(elevels, ef=ef, **kwargs)
            diff = (occs*weights).sum() - ncharge
            return diff

        emin = elevels[0] - 2*self.degauss
        emax = elevels[nbands] + 2*self.degauss
        ef = (emin + emax)/2.0
        step = min(1.0, (emax - ef)*(0.5+tol))

        lfirst = True
        for i in range(maxiter):
            diff = func(ef)
            # print('eff', ef, emin, emax, step, diff, ncharge)
            if diff * step > 0.0 :
                step *= -0.25
            if abs(diff) < tol: break
            if ef > emax or ef < emin :
                if lfirst :
                    # step = self.degauss*4.0*np.sign(step)
                    lfirst = False
                    if ef > emax : ef = emax
                    if ef < emin : ef = emin
                else :
                    raise AttributeError(f"Can not find the good fermi level. ({ef} -> {diff})")
            ef += step
        else :
            raise AttributeError(f"Can not find the good fermi level. ({ef} -> {diff})")

        return ef
