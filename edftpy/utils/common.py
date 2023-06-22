import numpy as np
from dftpy.functional.functional_output import FunctionalOutput as dftpy_func
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid, RadialGrid
from dftpy.ions import Ions
from abc import ABC, abstractmethod

def Grid(lattice, nr, direct = True, units=None, full=False, uppergrid = None, **kwargs):
    if hasattr(lattice, 'lattice') : lattice = lattice.lattice
    if direct :
        obj = DirectGrid(lattice, nr, units=units, full=full, uppergrid = uppergrid, **kwargs)
    else :
        obj = ReciprocalGrid(lattice, nr, units=units, full=full, uppergrid = uppergrid, **kwargs)
    return obj

def Field(grid, memo="", rank=1, data = None, direct = True, order = 'C', cplx = False):
        kwargs = {'memo' :memo, 'rank' :rank, 'cplx' :cplx, 'griddata_F' :None, 'griddata_C' :None}
        if data is None and not direct :
            data = np.zeros(grid.nr, dtype=np.complex128)
        if order == 'C' :
            kwargs['griddata_C'] = data
        else :
            kwargs['griddata_F'] = data

        if isinstance(grid, DirectGrid) :
            direct = True
        elif isinstance(grid, ReciprocalGrid) :
            direct = False
        else :
            raise AttributeError("Not support '{}' type grid".format(type(grid)))

        if direct :
            obj = DirectField(grid, **kwargs)
        else :
            obj = ReciprocalField(grid, **kwargs)
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
