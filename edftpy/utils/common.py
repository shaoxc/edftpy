import numpy as np
from dftpy.functional_output import Functional as dftpy_func
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.atom import Atom as dftpy_atom
from dftpy.base import DirectCell
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d, splrep, splev
from dftpy.math_utils import quartic_interpolation


def Grid(lattice, nr, direct = True, origin=np.array([0.0, 0.0, 0.0]), units=None, full=False, uppergrid = None, **kwargs):
    if direct :
        obj = DirectGrid(lattice, nr, origin = origin, units=units, full=full, uppergrid = uppergrid, **kwargs)
    else :
        obj = ReciprocalGrid(lattice, nr, origin = origin, units=units, full=full, uppergrid = uppergrid, **kwargs)
    return obj

class RadialGrid(object):
    def __init__(self, r = None, v = None, direct = True, **kwargs):
        self._r = r
        self._v = v
        self._v_interp = None
        self.direct = direct

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        self._r = r

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):
        self._v = v

    @property
    def v_interp(self):
        if self._v_interp is None :
            self._v_interp = splrep(self.r, self.v)
        return self._v_interp

    def to_3d_grid(self, dist, direct = None, out = None):
        if out is None :
            results = np.zeros_like(dist)
        else :
            results = out
        mask = dist < self._r[-1]
        results[mask] = splev(dist[mask], self.v_interp, der=0)
        #-----------------------------------------------------------------------
        mask = dist < self._r[1]
        v = self._v
        dp = v[1]-v[0]
        f = [v[2], v[1], v[0], v[1], v[2]]
        dx = dist[mask]/dp
        results[mask] = quartic_interpolation(f, dx)
        #-----------------------------------------------------------------------
        return results

def Field(grid, memo="", rank=1, data = None, direct = True, order = 'C', cplx = False):
        kwargs = {'memo' :memo, 'rank' :rank, 'cplx' :cplx, 'griddata_F' :None, 'griddata_C' :None}
        if data is None and not direct :
            data = np.zeros(grid.nr, dtype=np.complex128)
        if order == 'C' :
            kwargs['griddata_C'] = data
        else :
            kwargs['griddata_F'] = data
        if direct :
            obj = DirectField(grid, **kwargs)
        else :
            obj = ReciprocalField(grid, **kwargs)
        return obj

def Atoms(labels, zvals=None, pos=None, cell=None, basis="Cartesian", origin = [0.0, 0.0, 0.0], **kwargs):
    cell = DirectCell(cell, origin=origin)
    obj = dftpy_atom(Zval = zvals, label = labels, pos = pos, cell = cell)
    return obj

def Functional(name=None, energy=None, potential=None, **kwargs):
    obj = dftpy_func(name = name, energy = energy, potential = potential, **kwargs)
    return obj


class AbsFunctional(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, density, **kwargs):
        pass

    @abstractmethod
    def compute(self, density, **kwargs):
        pass


class AbsOptDriver(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, calcType = ['O', 'E', 'V'], **kwargs):
        pass

    @abstractmethod
    def compute(self, calcType = ['O', 'E', 'V'], **kwargs):
        pass


class AbsDFT(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_density(self, **kwargs):
        pass

    @abstractmethod
    def get_energy(self, **kwargs):
        pass

    @abstractmethod
    def update_density(self, **kwargs):
        pass

    @abstractmethod
    def get_energy_potential(self, **kwargs):
        pass

    @abstractmethod
    def get_fermi_level(self, **kwargs):
        pass
