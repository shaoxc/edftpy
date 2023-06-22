import numpy as np
from dftpy.math_utils import ecut2nr, bestFFTsize

from edftpy.functional import LocalPP, Ewald
from edftpy.utils.common import Field, Grid, Ions
from edftpy.density import get_3d_value_recipe, gen_gaussian_density
from edftpy.mpi import sprint

class SubCell(object):
    def __init__(self, ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], optfft = False, full = False, gaussian_options = None, nr = None, nspin = 1, **kwargs):
        self._grid = None
        self._ions = None
        self._density = None
        self._ions_index = index
        self.nspin = nspin

        self._gen_cell(ions, grid, index = index, cellcut = cellcut, optfft = optfft, full = full, nr = nr, **kwargs)
        self._density = Field(grid=self.grid, rank=self.nspin, direct=True)
        if gaussian_options is None or len(gaussian_options)==0:
            self._gaussian_density = None
        else :
            self._gen_gaussian_density(gaussian_options)
            # self._gen_gaussian_density_recip(gaussian_options)

        self.core_density = Field(grid=self.grid, rank=1, direct=True)

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

    @gaussian_density.setter
    def gaussian_density(self, value):
        self._gaussian_density = value

    @property
    def shift(self):
        return self.grid.shift

    def _gen_cell(self, ions, grid, index = None, grid_sub = None, **kwargs):

        if index is None : index = slice(None)
        pos_cry = ions.get_scaled_positions()[index]

        grid_sub = self.gen_grid_sub(ions, grid, index = index, grid_sub = grid_sub, **kwargs)

        origin = grid_sub.shift / grid.nrR
        pos_cry -= origin
        # pos_cry %= 1.0
        pos = ions.cell.cartesian_positions(pos_cry)

        ions_sub = Ions(numbers = ions.numbers[index].copy(), positions = pos, cell = grid_sub.lattice, charges = ions.charges[index].copy())
        self._grid = grid_sub
        self._ions = ions_sub
        self._ions_index = index
        self.comm = self._grid.mp.comm
        # sprint('subcell grid', self._grid.nrR, self._grid.nr, comm=self.comm)
        # sprint('subcell shift', self._grid.shift, comm=self.comm)

    @staticmethod
    def gen_grid_sub(ions, grid, index = None, cellcut = [0.0, 0.0, 0.0], cellsplit = None, optfft = True,
            full = False, grid_sub = None, max_prime = 5, scale = 1.0, nr = None, mp = None):
        tol = 1E-8
        cell = ions.cell
        lattice_sub = cell.copy()
        latp = cell.cellpar()[:3]
        if index is None : index = slice(None)
        if isinstance(cellcut, (int, float)) or len(cellcut) == 1 :
            cellcut = np.ones(3) * cellcut
        if cellsplit is not None :
            if isinstance(cellsplit, (int, float)) or len(cellsplit) == 1 :
                cellsplit = np.ones(3) * cellsplit
        spacings = grid.spacings.copy()
        shift = np.zeros(3, dtype = 'int')
        origin = np.zeros(3)
        if grid_sub is not None :
            nr = grid_sub.nrR

        pos_cry = ions.get_scaled_positions()[index]
        cs = np.min(pos_cry, axis = 0)
        pos_cry -= cs
        for i, p in enumerate(pos_cry) :
            for j in range(3):
                if pos_cry[i][j] > 0.5 :
                    pos_cry[i][j] -= 1.0
        pos = ions.cell.cartesian_positions(pos_cry)
        #-----------------------------------------------------------------------
        cell_size = np.ptp(pos, axis = 0)
        pbc = np.ones(3, dtype = 'int')

        if nr is None :
            nr = grid.nrR.copy()
            for i in range(3):
                if cellsplit is not None :
                    pbc[i] = False
                    if cellsplit[i] > (1.0-tol) :
                        origin[i] = 0.0
                        continue
                    cell_size[i] = cellsplit[i] * latp[i]
                    origin[i] = 0.5
                elif cellcut[i] > tol :
                    pbc[i] = False
                    cell_size[i] += cellcut[i] * 2.0
                    if cell_size[i] > (latp[i] - (max_prime + 1) * spacings[i]):
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
                    lattice_sub[i] *= (nr[i] * spacings[i]) / latp
        else :
            for i in range(3):
                if nr[i] < grid.nrR[i] :
                    lattice_sub[i] *= (nr[i] * spacings[i]) / latp[i]
                    origin[i] = 0.5
                else :
                    nr[i] = grid.nrR[i]
                    origin[i] = 0.0

        c1 = lattice_sub.cartesian_positions(origin)

        c0 = np.mean(pos, axis = 0)
        center = cell.scaled_positions(c0)
        center += cs
        center[origin < tol] = 0.0
        c0 = cell.cartesian_positions(center)

        origin = np.array(c0) - np.array(c1)
        shift[:] = np.array(cell.scaled_positions(origin)) * grid.nrR

        if grid_sub is None :
            grid_sub = Grid(lattice=lattice_sub, nr=nr, full=full, direct = True, origin = origin, pbc = pbc, mp = mp)
        grid_sub.shift = shift
        return grid_sub

    def _gen_gaussian_density(self, options={}):
        self._gaussian_density, ncharge = gen_gaussian_density(self.ions, self.grid, options=options)
        nc = self._gaussian_density.integral()
        sprint(f'fake : {nc : >16.8E} deviation : {nc - ncharge: >16.8E}', comm=self.comm)

        # if ncharge > 1E-3 :
            # factor = ncharge / (self._gaussian_density.integral())
        # else :
            # factor = 0.0
        # self._gaussian_density *= factor
        # nc = self._gaussian_density.integral()
        # sprint('fake02', nc, comm=self.comm)

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
                if self.ions.symbols[i] != key:
                    continue
                ncharge += scale
            r_g[key] = np.linspace(0, 30, 10000)
            arho_g[key] = np.exp(-2 * (sigma * r_g[key] * np.pi) ** 2) * scale
        self._gaussian_density[:]=get_3d_value_recipe(r_g, arho_g, self.ions, self.grid, ncharge = ncharge, direct=True, pme=True, order=10)
        nc = self._gaussian_density.integral()
        sprint('fake01', nc, comm=self.comm)

    def free(self):
        self.grid.free()


class GlobalCell(object):
    def __init__(self, ions, grid = None, ecut = 22, spacing = None, nr = None, full = False, optfft = True, graphtopo = None, nspin = 1, index = None, **kwargs):
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
        self.nspin = nspin
        self._ions_index = index

        self.restart(grid=grid, ions=ions)

    def restart(self, grid=None, ions=None):
        self._ions = ions
        self._grid = grid
        if self._grid is None :
            self._grid = self.gen_grid(self.ions.cell, **self.grid_kwargs)
            sprint('GlobalCell grid', self._grid.nrR, comm=self.comm)
        self._density = Field(grid=self.grid, rank=self.nspin, direct=True)
        self._gaussian_density = Field(grid=self.grid, rank=1, direct=True)
        self._core_density = Field(grid=self.grid, rank=1, direct=True)

    @property
    def ions_index(self):
        if self._ions_index is None :
            self._ions_index = slice(None)
        return self._ions_index

    @property
    def grid(self):
        return self._grid

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

    @staticmethod
    def gen_grid(lattice, ecut = 22, spacing = None, full = False, nr = None, optfft = True, max_prime = 5, scale = 1.0, mp = None, **kwargs):
        if nr is None :
            nr = ecut2nr(ecut, lattice, optfft = optfft, spacing = spacing, scale = scale, max_prime = max_prime)
        else :
            nr = np.asarray(nr)
        grid = Grid(lattice=lattice, nr=nr, full=full, direct = True, mp = mp, **kwargs)
        return grid

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

    def get_forces_old(self, linearii= True, **kwargs):
        for k, value in self.total_evaluator.funcdicts.items():
            if isinstance(value, LocalPP):
                forces_ie = value.force(self.density)
                break
        else :
            sprint('!WARN : There is no `LocalPP` in total_evaluator', comm=self.comm)
            forces_ie = np.zeros((self.ions.nat, 3))
        ewaldobj = Ewald(rho=self.density, ions=self.ions, PME=linearii)
        forces_ii = ewaldobj.forces
        forces = forces_ie + forces_ii
        return forces

    def get_forces(self, **kwargs):
        forces = np.zeros((self.ions.nat, 3))
        for k, value in self.total_evaluator.funcdicts.items():
            if hasattr(value, 'forces'):
                f = value.forces
                if hasattr(f, '__call__'):
                    f = f(self.density)
                if f is not None :
                    forces += f
        return forces

    def get_stress(self, **kwargs):
        stress = np.zeros((3, 3))
        for k, value in self.total_evaluator.funcdicts.items():
            if hasattr(value, 'stress'):
                f = value.stress
                if hasattr(f, '__call__'):
                    f = f(self.density)
                stress += f
        return stress
