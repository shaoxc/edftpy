import numpy as np
from scipy import signal
from abc import ABC, abstractmethod
import os

from dftpy.constants import ENERGY_CONV

from edftpy.mixer import PulayMixer, AbstractMixer
from edftpy.utils.common import Grid, Field, Functional
from edftpy.utils.math import grid_map_data
from edftpy.utils import clean_variables
from edftpy.functional import hartree_energy
from edftpy.mpi import sprint, MP


class Driver(ABC):
    def __init__(self, technique = 'OF', key = None,
            evaluator = None, subcell = None, prefix = 'sub_of', options = None, exttype = 3,
            mixer = None, ncharge = None, task = 'scf', append = False, nspin = 1,
            restart = False, base_in_file = None, **kwargs):
        '''
        Here, prefix is the name of the input file of the driver
        exttype :
                    1 : only pseudo                  : 001
                    2 : only hartree                 : 010
                    3 : hartree + pseudo             : 011
                    4 : only xc                      : 100
                    5 : pseudo + xc                  : 101
                    6 : hartree + xc                 : 110
                    7 : pseudo + hartree + xc        : 111
        '''
        self.technique = technique
        self.key = key # the key of config
        self.evaluator = evaluator
        self.subcell = subcell
        self.prefix = prefix
        self.exttype = exttype
        self.mixer = mixer
        self.ncharge = ncharge
        self.task = task
        self.append = append
        self.nspin = nspin
        self.restart = restart
        self.base_in_file = base_in_file
        self.filename = base_in_file
        default_options = {
                'update_delay' : 1,
                'update_freq' : 1
        }
        self.options = default_options
        if options is not None :
            self.options.update(options)
        self.comm = self.subcell.grid.mp.comm
        #-----------------------------------------------------------------------
        self.density = None
        self.potential = None
        self.gaussian_density = None
        self.core_density = None
        self.residual_norm = 0.0
        self.dp_norm = 0.0
        self.energy = None

    def get_density(self, **kwargs):
        pass

    def get_energy(self, **kwargs):
        pass

    def update_density(self, **kwargs):
        pass

    @abstractmethod
    def get_energy_potential(self, **kwargs):
        pass

    def get_fermi_level(self, **kwargs):
        return 0.0

    def update_workspace(self, *arg, **kwargs):
        pass

    def end_scf(self, **kwargs):
        pass

    def stop_run(self, *arg, **kwargs):
        pass

    def save(self, *arg, **kwargs):
        pass

    def __call__(self, density = None, gsystem = None, calcType = ['O', 'E'], **kwargs):
        return self.compute(density, gsystem, calcType, **kwargs)

    def compute(self, density = None, gsystem = None, calcType = ['O', 'E'], **kwargs):
        if 'O' in calcType :

            if gsystem is None and self.evaluator.gsystem is None :
                raise AttributeError("Must provide global system")
            else:
                self.evaluator.gsystem = gsystem

            self.get_density(**kwargs)
            self.mu = self.get_fermi_level()

        if 'E' in calcType or 'V' in calcType :
            func = self.get_energy_potential(self.density, calcType, **kwargs)
            self.functional = func

        if 'E' in calcType :
            self.energy = func.energy

        if 'V' in calcType :
            self.potential = func.potential
        return

    def get_gaussian_density(self, subcell, grid = None, **kwargs):
        if subcell.grid.mp.is_mpi and self.technique != 'OF' and subcell.gaussian_density is not None :
            gaussian_density = subcell.gaussian_density.gather(grid = grid)
        else :
            gaussian_density = subcell.gaussian_density
        return gaussian_density

    def _windows_function(self, grid, alpha = 0.5, bd = [5, 5, 5], **kwargs):
        wf = []
        for i in range(3):
            if grid.pbc[i] :
                wind = np.ones(grid.nr[i])
            else :
                wind = np.zeros(grid.nr[i])
                n = grid.nr[i] - 2 * bd[i]
                wind[bd[i]:n+bd[i]] = signal.tukey(n, alpha)
            wf.append(wind)
        array = np.einsum("i, j, k -> ijk", wf[0], wf[1], wf[2])
        return array

    @property
    def filter(self):
        if self._filter is None :
            self._filter = self._windows_function(self.grid)
        return self._filter

class DriverConstraint(object):
    """
    Give some constraint for the driver.

    Notes:
        Most time still use driver to do the thing.
    """
    # funcdicts = ['get_density', 'update_density', 'end_scf',
    #         'density', 'driver', 'residual_norm', 'dp_norm']

    def __init__(self, driver = None, density = None, **kwargs):
        self.driver = driver
        self.density = density
        if density is None :
            self.density = self.driver.density.copy()
        self.residual_norm = 0.0
        self.dp_norm = 0.0

    def get_density(self, **kwargs):
        return self.density

    def update_density(self, **kwargs):
        return self.density

    def end_scf(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        pass

    def __getattr__(self, attr):
        if attr in dir(self):
            return object.__getattribute__(self, attr)
        else :
            return getattr(self.driver, attr)

class Engine(ABC):
    """
    Note:
        embed : The object contains the embedding information.
        units : The engine/driver to eDFTpy. (e.g. energy : Ry -> Hartree is 0.5)
    """
    def __init__(self, units = {}, **kwargs):
        self.units = {
                'length' : 1.0,
                'volume' : 1.0,
                'energy' : 1.0,
                'order' : 'F',
                }
        self.units.update(units)

    def get_force(self, icalc = 0, **kwargs):
        force = None
        return force

    def embed_base(self, *args, **kwargs):
        embed = None
        return embed

    def calc_energy(self, **kwargs):
        energy = None
        return energy

    def clean_saved(self, *args, **kwargs):
        pass

    def forces(self, icalc = 0, **kwargs):
        pass

    @abstractmethod
    def get_grid(self, nr = None, **kwargs):
        pass

    def get_rho(self, rho = None, **kwargs):
        pass

    def get_rho_core(self, rho = None, **kwargs):
        pass

    def get_ef(self, **kwargs):
        return 0.0

    def initial(self, inputfile = None, comm = None, **kwargs):
        pass

    def save(self, save = ['D'], **kwargs):
        pass

    def scf(self, **kwargs):
        pass

    def scf_mix(self, **kwargs):
        pass

    def set_extpot(self, extpot, **kwargs):
        pass

    def set_rho(self, rho, **kwargs):
        pass

    def set_stdout(self, outfile, append = False, **kwargs):
        pass

    def stop_scf(self, status = 0, save = ['D'], **kwargs):
        pass

    def stop_tddft(self, status = 0, save = ['D'], **kwargs):
        pass

    def end_scf(self, **kwargs):
        pass

    def end_tddft(self, **kwargs):
        pass

    def tddft(self, **kwargs):
        pass

    def tddft_after_scf(self, inputfile = None, **kwargs):
        pass

    def tddft_initial(self, inputfile = None, comm = None, **kwargs):
        pass

    def update_ions(self, subcell, update = 0, **kwargs):
        # update = 0  all
        # update = 1  atomic configuration dependent information
        pass

    def wfc2rho(self, *args, **kwargs):
        pass

    def write_stdout(self, line, **kwargs):
        pass

    def write_input(self, filename = 'sub_driver.in', subcell = None, params = {}, cell_params = {}, base_in_file = None, **kwargs):
        pass

    def get_potential(self, **kwargs):
        return None

    def get_dnorm(self, **kwargs):
        return 0.0

class DriverKS(Driver):
    """
    Note :
        The potential and density will gather in rank == 0 for engine.
    """
    def __init__(self, engine = None, **kwargs):
        '''
        Here, prefix is the name of the input file
        exttype :
                    1 : only pseudo                  : 001
                    2 : only hartree                 : 010
                    3 : hartree + pseudo             : 011
                    4 : only xc                      : 100
                    5 : pseudo + xc                  : 101
                    6 : hartree + xc                 : 110
                    7 : pseudo + hartree + xc        : 111
        '''
        kwargs["technique"] = 'KS'
        super().__init__(**kwargs)
        self.engine = engine

        self._input_ext = '.in'
        if self.prefix :
            if self.comm.rank == 0 :
                self.build_input(**kwargs)
        else :
            self.prefix, self._input_ext= os.path.splitext(self.base_in_file)

        if self.comm.size > 1 : self.comm.Barrier()
        self._grid = None
        self._grid_sub = None
        self.outfile = self.prefix + '.out'
        self._driver_initialise(append = self.append)
        self.grid_driver = self.get_grid_driver(self.grid)
        #-----------------------------------------------------------------------
        self.atmp = np.zeros(1)
        self.atmp2 = np.zeros((1, self.nspin), order='F')
        #-----------------------------------------------------------------------
        self.mix_driver = None
        if self.mixer is None :
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, 0.6, 1.0], maxm = 7, coef = 0.5, predecut = 0, delay = 1)
        elif isinstance(self.mixer, float):
            self.mix_driver = self.mixer

        fstr = f'Subcell grid({self.prefix}): {self.subcell.grid.nrR}  {self.subcell.grid.nr}\n'
        fstr += f'Subcell shift({self.prefix}): {self.subcell.grid.shift}\n'
        if self.grid_driver is not None :
            fstr += f'{self.__class__.__name__} has two grids :{self.grid.nrR} and {self.grid_driver.nrR}'
        sprint(fstr, comm=self.comm, level=1)
        self.engine.write_stdout(fstr)
        #-----------------------------------------------------------------------
        self.init_density()
        self.embed = self.engine.embed_base(**kwargs)
        self.update_workspace(first = True, restart = self.restart)

    def update_workspace(self, subcell = None, first = False, update = 0, restart = False, **kwargs):
        """
        Notes:
            clean workspace
        """
        self.rho = None
        self.wfs = None
        self.occupations = None
        self.eigs = None
        self.fermi = None
        self.calc = None
        self._iter = 0
        self.energy = 0.0
        self.phi = None
        self.residual_norm = 1.0
        self.dp_norm = 1.0
        if isinstance(self.mixer, AbstractMixer):
            self.mixer.restart()
        if subcell is not None :
            self.subcell = subcell

        self.gaussian_density = self.get_gaussian_density(self.subcell, grid = self.grid)

        if not first :
            self.engine.update_ions(self.subcell, update)
            if update == 0 :
                # get new density
                self.engine.get_rho(self.charge)
                self.density[:] = self._format_density_invert()
        if self.task == 'optical' :
            if first :
                self.engine.tddft_after_scf(self.prefix + self._input_ext)
                if restart :
                    self.engine.wfc2rho()
                    # get new density
                    self.engine.get_rho(self.charge)
                    self.density[:] = self._format_density_invert()

        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        if self.comm.rank == 0 :
            core_charge = np.empty(grid.nnr, order = 'F')
        else :
            core_charge = self.atmp
        self.engine.get_rho_core(core_charge)
        self.core_density= self._format_density_invert(core_charge)

        self.core_density_sub = Field(grid = self.grid_sub)
        self.grid_sub.scatter(self.core_density, out = self.core_density_sub)
        if self.comm.rank == 0 :
            clean_variables(core_charge)
        return

    @property
    def grid(self):
        if self._grid is None :
            if np.all(self.subcell.grid.nrR == self.subcell.grid.nr):
                self._grid = self.subcell.grid
            else :
                self._grid = Grid(self.subcell.grid.lattice, self.subcell.grid.nrR, direct = True)
        return self._grid

    @property
    def grid_sub(self):
        if self._grid_sub is None :
            mp = MP(comm = self.comm, decomposition = self.subcell.grid.mp.decomposition)
            self._grid_sub = Grid(self.subcell.grid.lattice, self.subcell.grid.nrR, direct = True, mp = mp)
        return self._grid_sub

    def get_grid_driver(self, grid):
        nr = np.zeros(3, dtype = 'int32')
        self.engine.get_grid(nr)
        if not np.all(grid.nrR == nr) and self.comm.rank == 0 :
            grid_driver = Grid(grid.lattice, nr, direct = True)
        else :
            grid_driver = None
        return grid_driver

    def build_input(self, **kwargs):
        filename = self.prefix + self._input_ext
        self.engine.write_input(filename = filename, **kwargs)

    def _driver_initialise(self, append = False, **kwargs):
        comm = self.comm
        if self.comm.rank == 0 :
            self.engine.set_stdout(self.outfile, append = append)
        if self.task == 'optical' :
            self.engine.tddft_initial(self.prefix + self._input_ext, comm)
        else :
            self.engine.initial(self.prefix + self._input_ext, comm)

    def init_density(self, rho_ini = None):
        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        if self.comm.rank == 0 :
            self.density = Field(grid=self.grid)
            self.prev_density = Field(grid=self.grid)
            self.charge = np.empty((grid.nnr, self.nspin), order = 'F')
            self.prev_charge = np.empty((grid.nnr, self.nspin), order = 'F')
        else :
            self.density = self.atmp
            self.prev_density = self.atmp
            self.charge = self.atmp2
            self.prev_charge = self.atmp2

        self.density_sub = Field(grid = self.grid_sub)
        self.gaussian_density_sub = Field(grid = self.grid_sub)

        if rho_ini is None :
            nc = np.sum(self.subcell.density)
            if nc > -1E-6 :
                rho_ini = self.subcell.density.gather(grid = self.grid)

        if rho_ini is not None :
            if self.comm.rank == 0 : self.density[:] = rho_ini
            self._format_density()
        else :
            self.engine.get_rho(self.charge)
            self.density[:] = self._format_density_invert()

    def _format_density(self, volume = None, sym = False, **kwargs):
        self.prev_charge, self.charge = self.charge, self.prev_charge
        if self.grid_driver is not None and self.comm.rank == 0:
            charge = grid_map_data(self.density, grid = self.grid_driver)
        else :
            charge = self.density
        #-----------------------------------------------------------------------
        charge = charge.reshape((-1, self.nspin), order='F') / self.engine.units['volume']
        self.charge[:] = charge
        self.engine.set_rho(charge)
        #-----------------------------------------------------------------------
        self.prev_density[:] = self.density
        return

    def _format_density_invert(self, charge = None, grid = None, **kwargs):
        if self.comm.rank > 0 : return self.atmp
        if charge is None :
            charge = self.charge

        if grid is None :
            grid = self.grid

        if self.grid_driver is not None and np.any(self.grid_driver.nrR != grid.nrR):
            density = Field(grid=self.grid_driver, direct=True, data = charge, order = 'F')
            rho = grid_map_data(density, grid = grid)
        else :
            rho = Field(grid=grid, rank=1, direct=True, data = charge, order = 'F')
        rho *= self.engine.units['volume']
        return rho

    def _get_extpot_serial(self, with_global = True, **kwargs):
        if self.comm.rank == 0 :
            self.evaluator.get_embed_potential(self.density, gaussian_density = self.gaussian_density, with_global = with_global)
            extpot = self.evaluator.embed_potential
        else :
            extpot = self.atmp
        return extpot

    def _get_extpot_mpi(self, with_global = True, **kwargs):
        self.grid_sub.scatter(self.density, out = self.density_sub)
        if self.gaussian_density is not None :
            self.grid_sub.scatter(self.gaussian_density, out = self.gaussian_density_sub)
        self.evaluator.get_embed_potential(self.density_sub, gaussian_density = self.gaussian_density_sub, gather = True, with_global = with_global)
        extpot = self.evaluator.embed_potential
        return extpot

    def get_extpot(self, extpot = None, mapping = True, **kwargs):
        if extpot is None :
            # extpot = self._get_extpot_serial(**kwargs)
            extpot = self._get_extpot_mpi(**kwargs)
        if mapping :
            if self.grid_driver is not None and self.comm.rank == 0:
                extpot = grid_map_data(extpot, grid = self.grid_driver)
            extpot = extpot.ravel(order = 'F')
        return extpot

    def _get_extene(self, extpot, **kwargs):
        if self.comm.rank == 0 :
            extene = (extpot * self.density).integral()
        else :
            extene = 0.0

        if self.comm.size > 1 :
            extene = self.comm.bcast(extene, root = 0)
        return extene

    def _get_charge(self, **kwargs):
        if self.task == 'optical' :
            self.engine.tddft(**kwargs)
        else :
            self.engine.scf(**kwargs)

        self.engine.get_rho(self.charge)
        return self.charge

    def get_density(self, vext = None, sdft = 'sdft', **kwargs):
        self._iter += 1
        if self._iter == 1 :
            # The first step density mixing also need keep initial = True
            initial = True
        else :
            initial = False

        if self.mix_driver is not None :
            # If use pwscf mixer, do not need format the density
            self.prev_density[:] = self.density
        else :
            self._format_density()

        if sdft == 'pdft' :
            extpot = self.evaluator.embed_potential
            extpot = self.get_extpot(extpot, mapping = True)
            self.embed.lewald = False
        else :
            extpot = self.get_extpot()

        self.prev_charge[:] = self.charge

        self.engine.set_extpot(extpot)
        #
        self._get_charge(initial = initial)
        #
        self.energy = 0.0
        self.dp_norm = 1.0
        self.density[:] = self._format_density_invert()
        return self.density

    def get_energy(self, olevel = 0, sdft = 'sdft', **kwargs):
        if olevel == 0 :
            # self._format_density() # Here, we directly use saved density
            if sdft == 'pdft' :
                extpot = self.evaluator.embed_potential
                extpot = self.get_extpot(extpot, mapping = True)
            else :
                extpot = self.get_extpot()
            self.engine.set_extpot(extpot)
            energy = self.engine.calc_energy()
        else :
            energy = 0.0
        return energy

    def get_energy_potential(self, density = None, calcType = ['E', 'V'], olevel = 1, sdft = 'sdft', **kwargs):
        if 'E' in calcType :
            energy = self.get_energy(olevel = olevel, sdft = sdft)

            if olevel == 0 :
                self.grid_sub.scatter(density, out = self.density_sub)
                edict = self.evaluator(self.density_sub, calcType = ['E'], with_global = False, with_embed = False, gather = True, split = True)
                func = edict.pop('TOTAL')
            else : # elif olevel == 1 :
                if self.comm.rank == 0 :
                    func = self.evaluator(density, calcType = ['E'], with_global = False, with_embed = True)
                else :
                    func = Functional(name = 'ZERO', energy=0.0, potential=None)

            func.energy += energy
            if sdft == 'sdft' and self.exttype == 0 : func.energy = 0.0
            if self.comm.rank > 0 : func.energy = 0.0

            if olevel == 0 :
                fstr = format("Energy information", "-^80") + '\n'
                for key, item in edict.items():
                    fstr += "{:>12s} energy: {:22.15E} (eV) = {:22.15E} (a.u.)\n".format(key, item.energy* ENERGY_CONV["Hartree"]["eV"], item.energy)
                fstr += "-" * 80 + '\n'
            else :
                fstr = ''
            fstr += f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm, level=1)
            self.engine.write_stdout(fstr)
            self.energy = func.energy
        return func

    def update_density(self, coef = None, mix_grid = False, **kwargs):
        if self.comm.rank == 0 :
            if self.grid_driver is not None and mix_grid:
                prev_density = self._format_density_invert(self.prev_charge, self.grid_driver)
                density = self._format_density_invert(self.charge, self.grid_driver)
            else :
                prev_density = self.prev_density
                density = self.density
            #-----------------------------------------------------------------------
            r = density - prev_density
            self.residual_norm = np.sqrt(np.sum(r * r)/r.size)
            rmax = r.amax()
            fstr = f'res_norm({self.prefix}): {self._iter}  {rmax}  {self.residual_norm}'
            sprint(fstr, comm=self.comm, level=1)
            self.engine.write_stdout(fstr)
            #-----------------------------------------------------------------------
            if self.mix_driver is None :
                self.dp_norm = hartree_energy(r)
                rho = self.mixer(prev_density, density, **kwargs)
                if self.grid_driver is not None and mix_grid:
                    rho = grid_map_data(rho, grid = self.grid)
                self.density[:] = rho

        if self.mix_driver is not None :
            if coef is None : coef = self.mix_driver
            self.engine.scf_mix(coef = coef)
            self.dp_norm = self.engine.get_dnorm()
            self.engine.get_rho(self.charge)
            self.density[:] = self._format_density_invert()

        if self.comm.rank > 0 :
            self.residual_norm = 0.0
            self.dp_norm = 0.0
        # if self.comm.size > 1 : self.residual_norm = self.comm.bcast(self.residual_norm, root=0)
        return self.density

    def get_fermi_level(self, **kwargs):
        results = self.engine.get_ef()
        return results

    def get_forces(self, icalc = 2, **kwargs):
        """
        icalc :
                0 : all
                1 : no ewald
                2 : no ewald and local_potential
        """
        forces = self.engine.get_force()
        return forces

    def get_stress(self, **kwargs):
        pass

    def end_scf(self, **kwargs):
        if self.task == 'optical' :
            self.engine.end_tddft()
        else :
            self.engine.end_scf()

    def save(self, save = ['D'], **kwargs):
        self.engine.save(save)

    def stop_run(self, status = 0, save = ['D'], **kwargs):
        if self.task == 'optical' :
            self.engine.stop_tddft(status, save = save)
        else :
            self.engine.stop_scf(status, save = save)

class DriverEX(Driver):
    """
    Note :
        The potential and density will gather in rank == 0 for engine.
    """
    def __init__(self, engine = None, **kwargs):
        kwargs["technique"] = kwargs.get("technique", 'EX')
        super().__init__(**kwargs)
        self.engine = engine

        self.build_input(**kwargs)

        self.prefix, self._input_ext= os.path.splitext(self.base_in_file)

        if self.comm.size > 1 : self.comm.Barrier()
        self._grid = None
        self._driver_initialise(append = self.append)
        self.grid_driver = self.get_grid_driver(self.grid)
        #-----------------------------------------------------------------------
        self.atmp = np.zeros(1)
        self.atmp2 = np.zeros((1, self.nspin), order='F')
        #-----------------------------------------------------------------------
        self.mix_driver = None
        if self.mixer is None :
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, 0.6, 1.0], maxm = 7, coef = 0.5, predecut = 0, delay = 1)
        elif isinstance(self.mixer, float):
            self.mix_driver = self.mixer

        fstr = f'Subcell grid({self.prefix}): {self.subcell.grid.nrR}  {self.subcell.grid.nr}\n'
        fstr += f'Subcell shift({self.prefix}): {self.subcell.grid.shift}\n'
        if self.grid_driver is not None :
            fstr += f'{self.__class__.__name__} has two grids :{self.grid.nrR} and {self.grid_driver.nrR}'
        sprint(fstr, comm=self.comm, level=1)
        self.engine.write_stdout(fstr)
        #-----------------------------------------------------------------------
        self.embed = self.engine.embed_base(**kwargs)
        self.update_workspace(first = True, restart = self.restart)

    def update_workspace(self, subcell = None, first = False, update = 0, restart = False, **kwargs):
        """
        Notes:
            clean workspace
        """
        self.fermi = None
        self._iter = 0
        self.energy = 0.0
        self.residual_norm = 0.0
        self.dp_norm = 0.0
        return

    @property
    def grid(self):
        if self._grid is None :
            if np.all(self.subcell.grid.nrR == self.subcell.grid.nr):
                self._grid = self.subcell.grid
            else :
                self._grid = Grid(self.subcell.grid.lattice, self.subcell.grid.nrR, direct = True)
        return self._grid

    @property
    def grid_sub(self):
        if self._grid_sub is None :
            mp = MP(comm = self.comm, decomposition = self.subcell.grid.mp.decomposition)
            self._grid_sub = Grid(self.subcell.grid.lattice, self.subcell.grid.nrR, direct = True, mp = mp)
        return self._grid_sub

    def get_grid_driver(self, grid):
        nr = np.zeros(3, dtype = 'int32')
        self.engine.get_grid(nr)
        if not np.all(grid.nrR == nr) and self.comm.rank == 0 :
            grid_driver = Grid(grid.lattice, nr, direct = True)
        else :
            grid_driver = None
        return grid_driver

    def build_input(self, **kwargs):
        self.engine.write_input(**kwargs)

    def _driver_initialise(self, append = False, **kwargs):
        comm = self.comm
        if self.task == 'optical' :
            self.engine.tddft_initial(self.filename, comm = self.comm)
        else :
            self.engine.initial(self.filename, comm = comm)

    def _format_field(self, density, **kwargs):
        if self.grid_driver is not None and self.comm.rank == 0:
            charge = grid_map_data(density, grid = self.grid_driver)
        else :
            charge = density
        #-----------------------------------------------------------------------
        charge = charge.reshape((-1, self.nspin), order=self.engine.units['order']) / self.engine.units['volume']
        return charge

    def _format_field_invert(self, charge, grid = None, **kwargs):
        if grid is None :
            grid = self.grid

        if self.grid_driver is not None and np.any(self.grid_driver.nrR != grid.nrR):
            density = Field(grid=self.grid_driver, direct=True, data = charge, order = self.engine.units['order'])
            rho = grid_map_data(density, grid = grid)
        else :
            rho = Field(grid=grid, rank=1, direct=True, data = charge, order = self.engine.units['order'])
        rho *= self.engine.units['volume']
        return rho

    def get_energy(self, olevel = 0, **kwargs):
        if olevel == 0 :
            energy = self.engine.calc_energy()
        else :
            energy = 0.0
        return energy

    def get_energy_potential(self, density = None, calcType = ['E', 'V'], olevel = 1, **kwargs):
        func = Functional(name = 'ZERO', energy=0.0, potential=None)
        # self.engine.set_extpot(self.evaluator.global_potential, **kwargs)
        if 'V' in calcType :
            self._iter += 1
            rho = self._format_field(density)
            self.engine.scf(rho, **kwargs)
            pot = self.engine.get_potential(**kwargs)
            func.potential = self._format_field_invert(pot)
        if 'E' in calcType :
            energy = self.get_energy(olevel = olevel)
            func.energy = energy
            if self.comm.rank > 0 : func.energy = 0.0
            fstr = f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm, level=1)
            self.engine.write_stdout(fstr)
        return func

class DriverMM(DriverEX):
    """
    Note :
        The potential and density will gather in rank == 0 for engine.
    """
    def __init__(self, **kwargs):
        kwargs["technique"] = 'MM'
        super().__init__(**kwargs)

    def _driver_initialise(self, append = False, **kwargs):
        self.engine.initial(filename = self.filename, comm = self.comm,
                subcell = self.subcell, grid = self.grid)

    def get_energy_potential(self, density = None, calcType = ['E', 'V'], olevel = 1, **kwargs):
        func = Functional(name = 'ZERO', energy=0.0, potential=None)
        self.engine.set_extpot(self.evaluator.global_potential, **kwargs)
        if 'V' in calcType :
            pot = self.engine.get_potential(grid = self.grid_driver, **kwargs)
            func.potential = self._format_field(pot)
        if 'E' in calcType :
            energy = self.get_energy(olevel = olevel)
            func.energy = energy
            if self.comm.rank > 0 : func.energy = 0.0
            fstr = f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm, level=1)
            self.engine.write_stdout(fstr)
        return func
