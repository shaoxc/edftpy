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


class Grid(DirectField, ReciprocalField):
    def __new__(cls, lattice, nr, direct = True, origin = None, units=None, full=False, uppergrid = None, **kwargs):
        obj = super().__new__(cls, lattice, nr, origin = origin, units=units, full=full, uppergrid = uppergrid, **kwargs)
        obj.direct = direct
        return obj


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
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, rho, **kwargs):
        pass

    @abstractmethod
    def computeEnergyPotential(self, rho, **kwargs):
        pass

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    def assignName(self, name):
        self.name = name

    def assignType(self, type):
        self.type = type
