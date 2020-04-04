import numpy as np
from dftpy.functional_output import Functional as dftpy_func
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.atom import Atom as dftpy_atom
from dftpy.base import DirectCell
from abc import ABC, abstractmethod


def Grid(lattice, nr, direct = True, origin = None, units=None, full=False, uppergrid = None, **kwargs):
    if direct :
        obj = DirectGrid(lattice, nr, origin = origin, units=units, full=full, uppergrid = uppergrid, **kwargs)
    else :
        obj = ReciprocalGrid(lattice, nr, origin = origin, units=units, full=full, uppergrid = uppergrid, **kwargs)
    return obj

def Field(grid, memo="", rank=1, data = None, direct = True, order = 'C', cplx = False):
        kwargs = {'memo' :memo, 'rank' :rank, 'cplx' :cplx, 'griddata_F' :None, 'griddata_C' :None}
        if order == 'C' :
            kwargs['griddata_C'] = data
        else :
            kwargs['griddata_F'] = data
        if direct :
            obj = DirectField(grid, **kwargs)
        else :
            obj = ReciprocalGrid(grid, **kwargs)
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
