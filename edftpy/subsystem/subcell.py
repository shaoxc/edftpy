import numpy as np
import copy
from dftpy.math_utils import bestFFTsize

from edftpy.utils.common import Field, Grid, Atoms


class SubCell(object):
    def __init__(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], optfft = False, **kwargs):
        self._grid = None
        self._ions = None
        self._density = None

        self._gen_cell(ions, grid, index = index, cellcut = cellcut, optfft = optfft, **kwargs)
        self._density = Field(grid=self.grid, rank=1, direct=True)

    @property
    def grid(self):
        if self._grid is None:
            raise AttributeError("Must generate subcell firstly")
        return self._grid

    @property
    def ions(self):
        if self._ions is None:
            raise AttributeError("Must generate subcell firstly")
        return self._ions

    @property
    def density(self):
        if self._density is None:
            raise AttributeError("Must set density firstly")
        return self._density

    @property
    def shift(self):
        return self._shift

    def _gen_cell(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], optfft = False):
        lattice_sub = ions.pos.cell.lattice.copy()
        if index is None :
            index = np.ones(ions.nat, dtype = 'bool')
        pos = ions.pos.to_cart()[index].copy()
        spacings = grid.spacings.copy()
        shift = np.zeros(3, dtype = 'int32')
        nr = grid.nr.copy()

        origin = np.min(pos, axis = 0)
        cell_size = np.max(pos, axis = 0)-origin

        for i in range(3):
            latp = np.linalg.norm(lattice_sub[:, i])
            if cellcut[i] > 1E-6 and latp > cellcut[i]:
                cell_size[i] += cellcut[i] * 2.0
                origin[i] -= cellcut[i]
                shift[i] = int(origin[i]/spacings[i])
                origin[i] = shift[i] * spacings[i]
                nr[i] = int(cell_size[i]/spacings[i])
                #-----------------------------------------------------------------------
                if optfft :
                    nr[i] = bestFFTsize(nr[i])
                lattice_sub[:, i] *= (nr[i] * spacings[i]) / latp
            else :
                origin[i] = 0.0
        pos -= origin
        print('subcell grid', nr)

        ions_sub = Atoms(ions.labels[index].copy(), zvals =ions.Zval, pos=pos, cell = lattice_sub, basis = 'Cartesian', origin = origin)
        grid_sub = Grid(lattice=lattice_sub, nr=nr, full=False, direct = True, origin = origin)
        grid_sub.shift = shift
        self._grid = grid_sub
        self._ions = ions_sub
        self._shift = shift


class GlobalCell(object):
    def __init__(self, ions, grid = None, spacing = 0.4, full = False, optfft = True, **kwargs):
        self._ions = ions
        self._grid = grid

        if self._grid is None :
            self._gen_grid(spacing = spacing, full = full, optfft = optfft, **kwargs)
        self._density = Field(grid=self.grid, rank=1, direct=True)

    @property
    def grid(self):
        return self._grid

    @property
    def ions(self):
        return self._ions

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        self._density[:] = value

    @property
    def total_evaluator(self):
        if self._total_evaluator is None:
            raise AttributeError("Must given total_evaluator")
        return self._total_evaluator

    @total_evaluator.setter
    def total_evaluator(self, value):
        self._total_evaluator = value

    def _gen_grid(self, spacing = None, full = False, optfft = True, **kwargs):
        lattice = self.ions.pos.cell.lattice
        metric = np.dot(lattice.T, lattice)
        nr = np.zeros(3, dtype = 'int32')
        for i in range(3):
            nr[i] = int(np.sqrt(metric[i, i])/spacing)
            if optfft :
                nr[i] = bestFFTsize(nr[i])
        grid = Grid(lattice=lattice, nr=nr, full=full, direct = True)
        self._grid = grid

    def update_density(self, subrho, restart = False):
        indl = subrho.grid.shift
        nr_sub = subrho.grid.nr
        indr = indl + nr_sub
        if restart :
            # self._density[:] = 0.0
            self._density[:] = 1E-30
        self._density[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]] += subrho
        return self._density

    def sub_value(self, total, subrho):
        indl = subrho.grid.shift
        nr_sub = subrho.grid.nr
        indr = indl + nr_sub
        value = total[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]]
        return value 

    def set_density(self, subrho, restart = False):
        indl = subrho.grid.shift
        nr_sub = subrho.grid.nr
        indr = indl + nr_sub
        self._density[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]] = subrho
        return self._density
