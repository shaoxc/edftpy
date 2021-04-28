import numpy as np
from numpy import linalg as LA
from dftpy.math_utils import ecut2nr, bestFFTsize
from dftpy.ewald import ewald

from edftpy.functional import LocalPP
from edftpy.utils.common import Field, Grid, Atoms, Coord
from edftpy.utils.math import gaussian
from edftpy.density import get_3d_value_recipe
from edftpy.mpi import sprint

class SubCell(object):
    def __init__(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], optfft = False, full = False, gaussian_options = None, nr = None, **kwargs):
        self._grid = None
        self._ions = None
        self._density = None
        self._ions_index = None

        self._gen_cell(ions, grid, index = index, cellcut = cellcut, optfft = optfft, full = full, nr = nr, **kwargs)
        self._density = Field(grid=self.grid, rank=1, direct=True)
        if gaussian_options is None or len(gaussian_options)==0:
            self._gaussian_density = None
        else :
            self._gen_gaussian_density(gaussian_options)
            # self._gen_gaussian_density_recip(gaussian_options)

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
    def ions_index(self):
        return self._ions_index

    @property
    def gaussian_density(self):
        return self._gaussian_density

    @property
    def shift(self):
        return self.grid.shift

    def _gen_cell(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], cellsplit = None, optfft = True,
            full = False, grid_sub = None, max_prime = 5, scale = 1.0, nr = None, mp = None):
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
        spacings = grid.spacings.copy()
        shift = np.zeros(3, dtype = 'int')
        origin = np.zeros(3)
        #-----------------------------------------------------------------------
        # pos = ions.pos.to_cart()[index].copy()
        # print('pos', pos)
        pos_cry = ions.pos.to_crys()[index].copy()
        cs = np.mean(pos_cry, axis = 0)
        pos_cry -= cs
        for i, p in enumerate(pos_cry) :
            for j in range(3):
                if pos_cry[i][j] > 0.5 :
                    pos_cry[i][j] -= 1.0
                elif pos_cry[i][j] < -0.5 :
                    pos_cry[i][j] += 1.0
        pos = pos_cry.to_cart()
        #-----------------------------------------------------------------------
        cell_size = np.ptp(pos, axis = 0)
        pbc = np.ones(3, dtype = 'int')

        if nr is None :
            nr = grid.nrR.copy()
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
                        nr[i] = bestFFTsize(nr[i], scale = scale, max_prime = max_prime)
                        nr[i] = min(nr[i], grid.nrR[i])
                    lattice_sub[:, i] *= (nr[i] * spacings[i]) / latp
        else :
            for i in range(3):
                if nr[i] < grid.nrR[i] :
                    latp = np.linalg.norm(lattice_sub[:, i])
                    lattice_sub[:, i] *= (nr[i] * spacings[i]) / latp
                    origin[i] = 0.5
                else :
                    nr[i] = grid.nrR[i]
                    origin[i] = 0.0

        cs_d = cs.to_cart()

        c1 = Coord(origin, lattice_sub, basis = 'Crystal').to_cart()

        c0 = np.mean(pos, axis = 0)
        center = Coord(c0, lattice, basis = 'Cartesian').to_crys()
        center += cs
        center[origin < 1E-6] = 0.0
        c0 = Coord(center, lattice, basis = 'Crystal').to_cart()

        pos += cs_d
        origin = np.array(c0) - np.array(c1)
        shift[:] = np.array(Coord(origin, lattice, basis = 'Cartesian').to_crys()) * grid.nrR
        origin[:] = shift / grid.nrR
        origin[:] = Coord(origin, lattice, basis = 'Crystal').to_cart()
        pos -= origin

        ions_sub = Atoms(ions.labels[index].copy(), zvals =ions.Zval, pos=pos, cell = lattice_sub, basis = 'Cartesian', origin = origin)
        if grid_sub is None :
            grid_sub = Grid(lattice=lattice_sub, nr=nr, full=full, direct = True, origin = origin, pbc = pbc, mp = mp)
        grid_sub.shift = shift
        self._grid = grid_sub
        self._ions = ions_sub
        self._ions_index = index
        self.comm = self._grid.mp.comm
        # sprint('subcell grid', self._grid.nrR, self._grid.nr, comm=self.comm)
        # sprint('subcell shift', self._grid.shift, comm=self.comm)

    def _gen_gaussian_density(self, options={}):
        """
        FWHM : 2*np.sqrt(2.0*np.log(2)) = 2.354820
        """
        fwhm = 2.354820
        nr = self.grid.nrR
        dnr = (1.0/nr).reshape((3, 1))
        gaps = self.grid.spacings
        # latp = self.grid.latparas
        self._gaussian_density = Field(self.grid)
        ncharge = 0.0
        sigma_min = np.max(gaps) * 2 / fwhm
        for key, option in options.items() :
            rcut = option.get('rcut', 5.0)
            sigma = option.get('sigma', 0.3)
            scale = option.get('scale', 0.0)
            if scale is None or abs(scale) < 1E-10 :
                continue
            # print('sigma', np.max(gaps), fwhm * sigma, sigma_min)
            sigma = max(sigma, sigma_min)
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
                mask = self.get_array_mask(l123A)
                self._gaussian_density[l123A[0][mask], l123A[1][mask], l123A[2][mask]] += prho.ravel()[mask]
                ncharge += scale

        nc = self._gaussian_density.integral()
        sprint('fake : ', nc, ' error : ', nc - ncharge, comm=self.comm)

        # if ncharge > 1E-3 :
            # factor = ncharge / (self._gaussian_density.integral())
        # else :
            # factor = 0.0
        # self._gaussian_density *= factor
        # nc = self._gaussian_density.integral()
        # sprint('fake02', nc, comm=self.comm)

    def get_array_mask(self, l123A):
        if self.comm.size == 1 :
            return slice(None)
        offsets = self.grid.offsets.reshape((3, 1))
        nr = self.grid.nr
        #-----------------------------------------------------------------------
        l123A -= offsets
        mask = np.logical_and(l123A[0] > -1, l123A[0] < nr[0])
        mask1 = np.logical_and(l123A[1] > -1, l123A[1] < nr[1])
        np.logical_and(mask, mask1, out = mask)
        np.logical_and(l123A[2] > -1, l123A[2] < nr[2], out = mask1)
        np.logical_and(mask, mask1, out = mask)
        #-----------------------------------------------------------------------
        return mask

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
            arho_g[key] = np.exp(-2 * (sigma * r_g[key] * np.pi) ** 2) * scale
        self._gaussian_density[:]=get_3d_value_recipe(r_g, arho_g, self.ions, self.grid, ncharge = ncharge, direct=True, pme=True, order=10)
        nc = self._gaussian_density.integral()
        sprint('fake01', nc, comm=self.comm)

class GlobalCell(object):
    def __init__(self, ions, grid = None, ecut = 22, spacing = None, nr = None, full = False, optfft = True, graphtopo = None, **kwargs):
        self.grid_kwargs = {
                'ecut' : ecut,
                'spacing' : spacing,
                'nr' : nr,
                'full' : full,
                'optfft' : optfft,
                }
        self.grid_kwargs.update(kwargs)
        if graphtopo is None :
            from edftpy.mpi import graphtopo
        self.graphtopo = graphtopo
        self.comm = self.graphtopo.comm

        self.restart(grid=grid, ions=ions)

    def restart(self, grid=None, ions=None):
        self._ions = ions
        self._grid = grid
        self._ewald = None
        if self._grid is None :
            self._gen_grid(**self.grid_kwargs)
        self._density = Field(grid=self.grid, rank=1, direct=True)
        self._gaussian_density = Field(grid=self.grid, rank=1, direct=True)
        self._core_density = Field(grid=self.grid, rank=1, direct=True)

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
    def core_density(self):
        return self._core_density

    @core_density.setter
    def core_density(self, value):
        self._core_density[:] = value

    @property
    def total_evaluator(self):
        if self._total_evaluator is None:
            raise AttributeError("Must given total_evaluator")
        return self._total_evaluator

    @total_evaluator.setter
    def total_evaluator(self, value):
        self._total_evaluator = value

    @property
    def graphtopo(self):
        if self._graphtopo is None :
            raise AttributeError("Must give graphtopo")
        return self._graphtopo

    @graphtopo.setter
    def graphtopo(self, value):
        self._graphtopo = value

    def _gen_grid(self, ecut = 22, spacing = None, full = False, nr = None, optfft = True, max_prime = 5, scale = 1.0, mp = None, **kwargs):
        lattice = self.ions.pos.cell.lattice
        if nr is None :
            nr = ecut2nr(ecut, lattice, optfft = optfft, spacing = spacing, scale = scale, max_prime = max_prime)
        else :
            nr = np.asarray(nr)
        grid = Grid(lattice=lattice, nr=nr, full=full, direct = True, mp = mp, **kwargs)
        self._grid = grid
        sprint('GlobalCell grid', nr, comm=self.comm)

    def update_density(self, subrho, isub = None, grid = None, fake = False, core = False, overwrite = False, **kwargs):
        if fake :
            total = self._gaussian_density
        elif core :
            total = self._core_density
        else :
            total = self._density
        self.graphtopo.sub_to_global(subrho, total, isub = isub, grid = grid, overwrite = overwrite)

    def set_density(self, subrho, isub = None, grid = None, fake = False, core = False):
        self.update_density(subrho, isub = isub, grid = grid, overwrite = True, fake = fake, core = core)

    def add_to_sub(self, total, subrho, isub = None, grid = None, add = True):
        self.graphtopo.global_to_sub(total, subrho, isub = isub, grid = grid, add = add)

    def add_to_global(self, subrho, total, isub = None, grid = None, overwrite = False):
        self.graphtopo.sub_to_global(subrho, total, isub = isub, grid = grid, overwrite = overwrite)

    def sub_value(self, total, subrho, isub = None, grid = None, **kwargs):
        self.graphtopo.global_to_sub(total, subrho, isub = isub, grid = grid, **kwargs)

    @property
    def gaussian_density(self):
        return self._gaussian_density

    def get_forces(self, linearii= True, **kwargs):
        for k, value in self.total_evaluator.funcdicts.items():
            if isinstance(value, LocalPP):
                forces_ie = value.force(self.density)
                break
        else :
            sprint('!WARN : There is no `LocalPP` in total_evaluator', comm=self.comm)
            forces_ie = np.zeros((self.ions.nat, 3))
        ewaldobj = ewald(rho=self.density, ions=self.ions, PME=linearii)
        forces_ii = ewaldobj.forces
        forces = forces_ie + forces_ii
        # forces_ii = self.grid.mp.vsum(forces_ii)
        # forces_ie = self.grid.mp.vsum(forces_ie)
        # sprint('ewald\n', forces_ii)
        # sprint('ie\n', forces_ie)
        return forces

    def get_stress(self, **kwargs):
        pass
