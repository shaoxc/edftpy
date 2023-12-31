import numpy as np
from scipy import signal
from abc import ABC, abstractmethod

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
                'update_delay' : 0,
                'update_freq' : 1,
                'update_sleep' : 0
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
