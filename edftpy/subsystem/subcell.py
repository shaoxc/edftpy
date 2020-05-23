import numpy as np
import copy
from dftpy.math_utils import bestFFTsize, ecut2spacing

from edftpy.utils.common import Field, Grid, Atoms
from ..utils.math import gaussian
from numpy import linalg as LA


class SubCell(object):
    def __init__(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], optfft = False, fake_core_options = None, **kwargs):
        self._grid = None
        self._ions = None
        self._density = None

        self._gen_cell(ions, grid, index = index, cellcut = cellcut, optfft = optfft, **kwargs)
        self._density = Field(grid=self.grid, rank=1, direct=True)
        if fake_core_options is None or len(fake_core_options)==0:
            self._fake_core_density = None
        else :
            self._gen_fake_core_density(fake_core_options)
            print('fake density', np.sum(self._fake_core_density) * self.grid.dV)

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
    def fake_core_density(self):
        return self._fake_core_density

    @property
    def shift(self):
        return self._shift

    def _gen_cell(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], optfft = True, full = False):
        lattice_sub = ions.pos.cell.lattice.copy()
        if index is None :
            index = np.ones(ions.nat, dtype = 'bool')
        pos = ions.pos.to_cart()[index].copy()
        spacings = grid.spacings.copy()
        shift = np.zeros(3, dtype = 'int32')
        nr = grid.nr.copy()

        origin = np.min(pos, axis = 0)
        cell_size = np.max(pos, axis = 0)-origin

        pbc = np.ones(3, dtype = 'int32')
        for i in range(3):
            latp = np.linalg.norm(lattice_sub[:, i])
            if cellcut[i] > 1E-6 and latp > 2.0 * cellcut[i]:
                pbc[i] = False
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
        grid_sub = Grid(lattice=lattice_sub, nr=nr, full=False, direct = True, origin = origin, pbc = pbc)
        grid_sub.shift = shift
        self._grid = grid_sub
        self._ions = ions_sub
        self._shift = shift

    def _gen_fake_core_density(self, options={}):
        nr = self.grid.nr
        dnr = (1.0/nr).reshape((3, 1))
        gaps = self.grid.spacings
        self._fake_core_density = Field(self.grid)
        for key, option in options.items() :
            rcut = option.get('rcut', 5.0)
            sigma = option.get('sigma', 0.4)
            scale = option.get('scale', 1.0)
            border = (rcut / gaps).astype(np.int32) + 1
            ixyzA = np.mgrid[-border[0]:border[0]+1, -border[1]:border[1]+1, -border[2]:border[2]+1].reshape((3, -1))
            prho = np.zeros((2 * border[0]+1, 2 * border[1]+1, 2 * border[2]+1))
            for i in range(self.ions.nat):
                if self.ions.labels[i] != key:
                    continue
                posi = self.ions.pos[i].reshape((1, 3))
                atomp = np.array(posi.to_crys()) * nr
                atomp = atomp.reshape((3, 1))
                ipoint = np.floor(atomp) 
                px = atomp - ipoint
                l123A = np.mod(ipoint.astype(np.int32) - ixyzA, nr[:, None])
                positions = (ixyzA + px) * dnr
                positions = np.einsum("j...,kj->k...", positions, self.grid.lattice)
                dists = LA.norm(positions, axis = 0).reshape(prho.shape)
                index = dists < rcut
                prho[index] = gaussian(dists[index], sigma) * scale
                # prho[index] = gaussian(dists[index], sigma, dim = 0) * scale
                self._fake_core_density[l123A[0], l123A[1], l123A[2]] += prho.ravel()

        ncharge = 0.0
        for i in range(self.ions.nat) :
            ncharge += self.ions.Zval[self.ions.labels[i]]
        factor = ncharge / (np.sum(self._fake_core_density) * self.grid.dV)
        self._fake_core_density *= factor

class GlobalCell(object):
    def __init__(self, ions, grid = None, ecut = 22, spacing = None, full = False, optfft = True, **kwargs):
        self._ions = ions
        self._grid = grid

        if self._grid is None :
            self._gen_grid(ecut = ecut, spacing = spacing, full = full, optfft = optfft, **kwargs)
        self._density = Field(grid=self.grid, rank=1, direct=True)
        self._fake_core_density = Field(grid=self.grid, rank=1, direct=True)

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

    def _gen_grid(self, ecut = 22, spacing = None, full = False, optfft = True, **kwargs):
        lattice = self.ions.pos.cell.lattice
        if spacing is None :
            spacing = ecut2spacing(ecut)
        metric = np.dot(lattice.T, lattice)
        nr = np.zeros(3, dtype = 'int32')
        for i in range(3):
            nr[i] = int(np.sqrt(metric[i, i])/spacing)
            if optfft :
                nr[i] = bestFFTsize(nr[i])
        grid = Grid(lattice=lattice, nr=nr, full=full, direct = True, **kwargs)
        self._grid = grid

    def update_density(self, subrho, grid = None, restart = False, fake = False):
        indl, indr = self.get_boundary(subrho, grid)
        if fake :
            if restart :
                self._fake_core_density[:] = 1E-30
            self._fake_core_density[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]] += subrho
        else :
            if restart :
                self._density[:] = 1E-30
            self._density[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]] += subrho
        return self._density

    def sub_value(self, total, subrho, grid = None):
        indl, indr = self.get_boundary(subrho, grid)
        value = total[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]].copy()
        return value 

    def set_density(self, subrho, grid = None, restart = False, fake = False):
        indl, indr = self.get_boundary(subrho, grid)
        if fake :
            self._fake_core_density[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]] += subrho
            return self._fake_core_density
        else :
            self._density[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]] = subrho
            return self._density

    def get_boundary(self, subrho, grid = None):
        if grid is None :
            grid = subrho.grid
        indl = grid.shift
        nr_sub = grid.nr
        indr = indl + nr_sub
        return indl, indr

    @property
    def fake_core_density(self):
        return self._fake_core_density
