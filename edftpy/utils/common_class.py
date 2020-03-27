import numpy as np
from dftpy.functional_output import Functional as dftpy_func
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.atom import Atom as dftpy_atom
from dftpy.base import DirectCell
from abc import ABC, abstractmethod


class Field(DirectField, ReciprocalField):
    def __new__(cls, grid, memo="", rank=1, data = None, direct = True, order = 'C', cplx = False):
        kwargs = {'memo' :memo, 'rank' :rank, 'cplx' :cplx, 'griddata_F' :None, 'griddata_C' :None}
        if order == 'C' :
            kwargs['griddata_C'] = data
        else :
            kwargs['griddata_F'] = data
        obj = super().__new__(cls, grid, **kwargs)
        obj.direct = direct
        return obj

    def integral(self):
        if self.direct :
            return DirectField.integral(self)
        else :
            return ReciprocalField.integral(self)

    def fft(self, **kwargs):
        #-----------------------------------------------------------------------
        self.direct = True
        #-----------------------------------------------------------------------
        if self.direct :
            results = DirectField.fft(self)
            results.direct = False
        else :
            results = ReciprocalField.ifft(self, **kwargs)
            results.direct = True
        return results

    def ifft(self, **kwargs):
        #-----------------------------------------------------------------------
        self.direct = False
        #-----------------------------------------------------------------------
        if self.direct :
            raise ValueError("Only ReciprocalField can do invert fft")
        else :
            results = ReciprocalField.ifft(self, **kwargs)
            results.direct = True
        return results


class Grid(DirectGrid, ReciprocalGrid):
    def __init__(self, lattice, nr, direct = True, origin = None, units=None, full=False, uppergrid = None, **kwargs):
        super().__init__(lattice, nr, origin = origin, units=units, full=full, uppergrid = uppergrid, **kwargs)
        self.direct = direct

    def get_direct(self, **kwargs):
        if self.direct :
            raise ValueError("Only ReciprocalGrid can get DirectGrid")
        else :
            results = ReciprocalGrid.get_direct(self, **kwargs)
            results.direct = True
        return results

    def get_reciprocal(self, **kwargs):
        if not self.direct :
            raise ValueError("Only DirectGrid can get ReciprocalGrid")
        else :
            results = DirectGrid.get_reciprocal(self, **kwargs)
            results.direct = False
        return results


class Atoms(dftpy_atom):
    def __new__(cls, labels, zvals=None, pos=None, cell=None, basis="Cartesian", **kwargs):
        cell = DirectCell(cell)
        obj = super().__new__(cls, Zval = zvals, label = labels, pos = pos, cell = cell)
        return obj


class Functional(dftpy_func):
    def __new__(cls, name=None, energy=None, potential=None, **kwargs):
        obj = super().__new__(cls, name = name, energy = energy, potential = potential, **kwargs)
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
