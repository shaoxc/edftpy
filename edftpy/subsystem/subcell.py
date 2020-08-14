import numpy as np
from numpy import linalg as LA
from dftpy.math_utils import ecut2nr, bestFFTsize
from dftpy.ewald import ewald

from edftpy.pseudopotential import LocalPP
from edftpy.utils.common import Field, Grid, Atoms, Coord
from ..utils.math import gaussian
from ..density import get_3d_value_recipe


class SubCell(object):
    def __init__(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], optfft = False, full = False, gaussian_options = None, coarse_grid_ecut = None, **kwargs):
        self._grid = None
        self._ions = None
        self._density = None
        self._grid_coarse = None
        self._density_coarse = None
        self._index_coarse = None
        self._ions_index = None

        self._gen_cell(ions, grid, index = index, cellcut = cellcut, optfft = optfft, full = full, coarse_grid_ecut = coarse_grid_ecut, **kwargs)
        self._density = Field(grid=self.grid, rank=1, direct=True)
        if gaussian_options is None or len(gaussian_options)==0:
            self._gaussian_density = None
        else :
            self._gen_gaussian_density(gaussian_options)
            # self._gen_gaussian_density_recip(gaussian_options)
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

    @ions.setter
    def ions(self, value):
        self._ions = value

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

    @property
    def ions_index(self):
        return self._ions_index

    @density_coarse.setter
    def density_coarse(self, value):
        self._density_coarse[:] = value

    @property
    def gaussian_density(self):
        return self._gaussian_density

    @property
    def shift(self):
        return self.grid.shift

    def _gen_cell(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], cellsplit = None, optfft = True, full = False, coarse_grid_ecut = None, grid_sub = None):
        tol = 1E-8
        lattice = ions.pos.cell.lattice.copy()
        lattice_sub = ions.pos.cell.lattice.copy()
        if index is None :
            index = np.ones(ions.nat, dtype = 'bool')
        if isinstance(cellcut, (int, float)) or len(cellcut) == 1 :
            cellcut = np.ones(3) * cellcut
        if cellsplit is not None :
            if isinstance(cellsplit, (int, float)) or len(cellsplit) == 1 :
                cellsplit = np.ones(3) * cellsplit
        pos = ions.pos.to_cart()[index].copy()
        spacings = grid.spacings.copy()
        shift = np.zeros(3, dtype = 'int')
        origin = np.zeros(3)
        nr = grid.nr.copy()

        cell_size = np.ptp(pos, axis = 0)
        pbc = np.ones(3, dtype = 'int')
        max_prime = 5
        for i in range(3):
            latp0 = np.linalg.norm(lattice[:, i])
            latp = np.linalg.norm(lattice_sub[:, i])
            if cellsplit is not None :
                pbc[i] = False
                if cellsplit[i] > (1.0-tol) :
                    origin[i] = 0.0
                    continue
                cell_size[i] = cellsplit[i] * latp0
                origin[i] = 0.5
            elif cellcut[i] > 1E-6 :
                pbc[i] = False
                cell_size[i] += cellcut[i] * 2.0
                if cell_size[i] > (latp0 - (max_prime + 1) * spacings[i]):
                    origin[i] = 0.0
                    continue
                origin[i] = 0.5
            else :
                origin[i] = 0.0

            if origin[i] > 0.01 :
                nr[i] = int(cell_size[i]/spacings[i])
                if optfft :
                    nr[i] = bestFFTsize(nr[i], scale = 1, max_prime = max_prime)
                    nr[i] = min(nr[i], grid.nr[i])
                lattice_sub[:, i] *= (nr[i] * spacings[i]) / latp

        c1 = Coord(origin, lattice_sub, basis = 'Crystal').to_cart()

        c0 = np.mean(pos, axis = 0)
        center = Coord(c0, lattice, basis = 'Cartesian').to_crys()
        center[origin < 1E-6] = 0.0
        c0 = Coord(center, lattice, basis = 'Crystal').to_cart()

        origin = np.array(c0) - np.array(c1)
        shift[:] = np.array(Coord(origin, lattice, basis = 'Cartesian').to_crys()) * grid.nr
        origin[:] = shift / grid.nr
        origin[:] = Coord(origin, lattice, basis = 'Crystal').to_cart()
        pos -= origin
        print('subcell grid', nr)
        print('subcell shift', shift)

        ions_sub = Atoms(ions.labels[index].copy(), zvals =ions.Zval, pos=pos, cell = lattice_sub, basis = 'Cartesian', origin = origin)
        if grid_sub is None :
            grid_sub = Grid(lattice=lattice_sub, nr=nr, full=full, direct = True, origin = origin, pbc = pbc)
        grid_sub.shift = shift
        self._grid = grid_sub
        self._ions = ions_sub
        self._ions_index = index
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
            scale = option.get('scale', 0.0)
            # scale = option.get('scale', self.ions.Zval[key])
            if scale is None or abs(scale) < 1E-10 :
                continue
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

    def _gen_gaussian_density_recip(self, options={}):
        self._gaussian_density = Field(self.grid)
        ncharge = 0.0
        r_g = {}
        arho_g = {}
        for key, option in options.items() :
            sigma = option.get('sigma', 0.3)
            scale = option.get('scale', 0.0)
            if scale is None or abs(scale) < 1E-10 :
                continue
            for i in range(self.ions.nat):
                if self.ions.labels[i] != key:
                    continue
                ncharge += scale
            r_g[key] = np.linspace(0, 30, 10000)
            arho_g[key] = np.exp(-0.5 * sigma ** 2 * r_g[key] ** 2) * scale
        self._gaussian_density[:]=get_3d_value_recipe(r_g, arho_g, self.ions, self.grid, ncharge = ncharge, direct=True, pme=True, order=10)
        print('fake01', np.sum(self._gaussian_density) * self.grid.dV)

class GlobalCell(object):
    def __init__(self, ions, grid = None, ecut = 22, spacing = None, nr = None, full = False, optfft = True, **kwargs):
        self.grid_kwargs = {
                'ecut' : ecut,
                'spacing' : spacing,
                'nr' : nr,
                'full' : full,
                'optfft' : optfft,
                }
        self.grid_kwargs.update(kwargs)

        self.restart(grid=grid, ions=ions)

    def restart(self, grid=None, ions=None):
        self._ions = ions
        self._grid = grid
        self._ewald = None
        if self._grid is None :
            self._gen_grid(**self.grid_kwargs)
        self._density = Field(grid=self.grid, rank=1, direct=True)
        self._gaussian_density = Field(grid=self.grid, rank=1, direct=True)

    @property
    def grid(self):
        return self._grid

    @property
    def ewald(self):
        if self._ewald is None :
            self._ewald = ewald(rho=self.density, ions=self.ions, PME=True)
        return self._ewald

    @property
    def ions(self):
        return self._ions

    @ions.setter
    def ions(self, value):
        self._ions = value

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
            nr = ecut2nr(ecut, lattice, optfft = optfft, spacing = spacing, scale = 1, max_prime = 5)
        else :
            nr = np.asarray(nr)
        grid = Grid(lattice=lattice, nr=nr, full=full, direct = True, **kwargs)
        self._grid = grid
        print('GlobalCell grid', nr)

    def update_density(self, subrho, grid = None, restart = False, fake = False, index = None, tol = 0.0):
        if index is None :
            index = self.get_sub_index(subrho, grid)
        if fake :
            if restart :
                self._gaussian_density[:] = tol
            self._gaussian_density[index[0], index[1], index[2]] += subrho.ravel()
        else :
            if restart :
                self._density[:] = tol
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

    def update_density_bound(self, subrho, grid = None, restart = False, fake = False, tol = 0.0):
        indl, indr = self.get_boundary(subrho, grid)
        if fake :
            if restart :
                self._gaussian_density[:] = tol
            self._gaussian_density[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]] += subrho
        else :
            if restart :
                self._density[:] = tol
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

    def get_forces(self, linearii= True, **kwargs):
        for k, value in self.total_evaluator.funcdicts.items():
            if isinstance(value, LocalPP):
                forces_ie = value.force(self.density)
                break
        else :
            print('!WARN : There is no `LocalPP` in total_evaluator')
            forces_ie = 0.0
        ewaldobj = ewald(rho=self.density, ions=self.ions, PME=linearii)
        forces_ii = ewaldobj.forces
        forces = forces_ie + forces_ii
        # print('ewald', forces_ii)
        # print('ie', forces_ie)
        return forces

    def get_stress(self, **kwargs):
        pass
