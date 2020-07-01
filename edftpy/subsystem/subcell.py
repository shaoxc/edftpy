import numpy as np
import copy
from dftpy.math_utils import ecut2nr, bestFFTsize

from edftpy.utils.common import Field, Grid, Atoms
from ..utils.math import gaussian
from numpy import linalg as LA


class SubCell(object):
    def __init__(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], optfft = False, full = False, gaussian_options = None, coarse_grid_ecut = None, **kwargs):
        self._grid = None
        self._ions = None
        self._density = None
        self._grid_coarse = None
        self._density_coarse = None
        self._index_coarse = None

        self._gen_cell(ions, grid, index = index, cellcut = cellcut, optfft = optfft, full = full, coarse_grid_ecut = coarse_grid_ecut, **kwargs)
        self._density = Field(grid=self.grid, rank=1, direct=True)
        if gaussian_options is None or len(gaussian_options)==0:
            self._gaussian_density = None
        else :
            self._gen_gaussian_density(gaussian_options)
            print('gaussian density', np.sum(self._gaussian_density) * self.grid.dV)

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

    @density.setter
    def density(self, value):
        if self._density is None:
            raise AttributeError("Must set density firstly")
        self._density[:] = value

    @property
    def grid_coarse(self):
        return self._grid_coarse

    @property
    def density_coarse(self):
        return self._density_coarse

    @property
    def index_coarse(self):
        return self._index_coarse

    @density_coarse.setter
    def density_coarse(self, value):
        self._density_coarse[:] = value

    @property
    def gaussian_density(self):
        return self._gaussian_density

    @property
    def shift(self):
        return self._shift

    def _gen_cell(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], optfft = True, full = False, coarse_grid_ecut = None):
        lattice = ions.pos.cell.lattice.copy()
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
            latp0 = np.linalg.norm(lattice[:, i])
            latp = np.linalg.norm(lattice_sub[:, i])
            if cellcut[i] > 1E-6 :
                pbc[i] = False
                cell_size[i] += cellcut[i] * 2.0
                if cell_size[i] > (latp0 - spacings[i]):
                    origin[i] = 0.0
                    continue
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
        grid_sub = Grid(lattice=lattice_sub, nr=nr, full=full, direct = True, origin = origin, pbc = pbc)
        grid_sub.shift = shift
        self._grid = grid_sub
        self._ions = ions_sub
        self._shift = shift
        if coarse_grid_ecut is not None :
            nr_coarse = ecut2nr(coarse_grid_ecut, lattice_sub, optfft = optfft)
            grid_coarse = Grid(lattice=lattice_sub, nr=nr_coarse, full=full, direct = True, origin = origin, pbc = pbc)
            self._grid_coarse = grid_coarse

    def _gen_gaussian_density(self, options={}):
        nr = self.grid.nr
        dnr = (1.0/nr).reshape((3, 1))
        gaps = self.grid.spacings
        # latp = self.grid.latparas
        self._gaussian_density = Field(self.grid)
        ncharge = 0.0
        for key, option in options.items() :
            rcut = option.get('rcut', 5.0)
            sigma = option.get('sigma', 0.4)
            scale = option.get('scale', 1.0)
            # scale = option.get('scale', self.ions.Zval[key])
            border = (rcut / gaps).astype(np.int32) + 1
            border = np.minimum(border, nr//2)
            ixyzA = np.mgrid[-border[0]:border[0]+1, -border[1]:border[1]+1, -border[2]:border[2]+1].reshape((3, -1))
            prho = np.zeros((2 * border[0]+1, 2 * border[1]+1, 2 * border[2]+1))
            for i in range(self.ions.nat):
                if self.ions.labels[i] != key:
                    continue
                prho[:] = 0.0
                posi = self.ions.pos[i].reshape((1, 3))
                atomp = np.array(posi.to_crys()) * nr
                atomp = atomp.reshape((3, 1))
                ipoint = np.floor(atomp + 1E-8) 
                # ipoint = np.floor(atomp)
                px = atomp - ipoint
                l123A = np.mod(ipoint.astype(np.int32) - ixyzA, nr[:, None])
                positions = (ixyzA + px) * dnr
                positions = np.einsum("j...,kj->k...", positions, self.grid.lattice)
                dists = LA.norm(positions, axis = 0).reshape(prho.shape)
                index = dists < rcut
                prho[index] = gaussian(dists[index], sigma) * scale
                # prho[index] = gaussian(dists[index], sigma, dim = 0) * scale
                self._gaussian_density[l123A[0], l123A[1], l123A[2]] += prho.ravel()
                ncharge += scale

        if ncharge > 1E-3 :
            factor = ncharge / (np.sum(self._gaussian_density) * self.grid.dV)
        else :
            factor = 0.0
        print('fake01', np.sum(self._gaussian_density) * self.grid.dV)
        self._gaussian_density *= factor
        print('fake02', np.sum(self._gaussian_density) * self.grid.dV)
        # rho_g = self._gaussian_density.fft()
        # self._gaussian_density = rho_g.ifft()

class GlobalCell(object):
    def __init__(self, ions, grid = None, ecut = 22, spacing = None, nr = None, full = False, optfft = True, **kwargs):
        self._ions = ions
        self._grid = grid

        if self._grid is None :
            self._gen_grid(ecut = ecut, spacing = spacing, nr = nr, full = full, optfft = optfft, **kwargs)
        self._density = Field(grid=self.grid, rank=1, direct=True)
        self._gaussian_density = Field(grid=self.grid, rank=1, direct=True)

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

    def _gen_grid(self, ecut = 22, spacing = None, full = False, nr = None, optfft = True, **kwargs):
        lattice = self.ions.pos.cell.lattice
        if nr is None :
            nr = ecut2nr(ecut, lattice, optfft = optfft, spacing = spacing)
        else :
            nr = np.asarray(nr)
        grid = Grid(lattice=lattice, nr=nr, full=full, direct = True, **kwargs)
        self._grid = grid
        print('GlobalCell grid', nr)

    def update_density(self, subrho, grid = None, restart = False, fake = False, index = None):
        if index is None :
            index = self.get_sub_index(subrho, grid)
        if fake :
            if restart :
                self._gaussian_density[:] = 1E-30
            self._gaussian_density[index[0], index[1], index[2]] += subrho.ravel()
        else :
            if restart :
                self._density[:] = 1E-30
            self._density[index[0], index[1], index[2]] += subrho.ravel()
        return self._density

    def sub_value(self, total, subrho, grid = None, index = None):
        if grid is None :
            if hasattr(subrho, 'grid'):
                grid = subrho.grid
            else :
                grid = subrho
        if index is None :
            index = self.get_sub_index(subrho, grid = grid)
        value = total[index[0], index[1], index[2]]
        value = Field(grid=grid, data=value, direct=True)
        return value 

    def set_density(self, subrho, grid = None, restart = False, fake = False, index = None):
        if index is None :
            index = self.get_sub_index(subrho, grid)
        if fake :
            self._gaussian_density[index[0], index[1], index[2]] = subrho.ravel()
            return self._gaussian_density
        else :
            self._density[index[0], index[1], index[2]] = subrho.ravel()
            return self._density

    def update_density_bound(self, subrho, grid = None, restart = False, fake = False):
        indl, indr = self.get_boundary(subrho, grid)
        if fake :
            if restart :
                self._gaussian_density[:] = 1E-30
            self._gaussian_density[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]] += subrho
        else :
            if restart :
                self._density[:] = 1E-30
            self._density[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]] += subrho
        return self._density

    def sub_value_bound(self, total, subrho, grid = None):
        indl, indr = self.get_boundary(subrho, grid)
        value = total[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]].copy()
        return value 

    def set_density_bound(self, subrho, grid = None, restart = False, fake = False):
        indl, indr = self.get_boundary(subrho, grid)
        if fake :
            self._gaussian_density[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]] = subrho
            return self._gaussian_density
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

    def get_sub_index(self, subrho = None, grid = None):
        if grid is None :
            if hasattr(subrho, 'grid'):
                grid = subrho.grid
            else :
                grid = subrho
        indl = grid.shift
        nr_sub = grid.nr
        indr = indl + nr_sub
        index = np.mgrid[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]].reshape((3, -1))
        nr = self.grid.nr
        for i in range(3):
            mask = index[i] < 0
            index[i][mask] += nr[i]
            mask2 = index[i] > nr[i] - 1
            index[i][mask2] -= nr[i]
        return index

    @property
    def gaussian_density(self):
        return self._gaussian_density
