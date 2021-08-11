import numpy as np
from scipy import signal
from abc import ABC, abstractmethod
from edftpy.utils.common import Field

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
        default_options = {
                'update_delay' : 1,
                'update_freq' : 1
        }
        self.options = default_options
        if options is not None :
            self.options.update(options)
        self.comm = self.subcell.grid.mp.comm

    @abstractmethod
    def get_density(self, **kwargs):
        pass

    @abstractmethod
    def get_energy(self, **kwargs):
        pass

    @abstractmethod
    def update_density(self, **kwargs):
        pass

    @abstractmethod
    def get_energy_potential(self, **kwargs):
        pass

    @abstractmethod
    def get_fermi_level(self, **kwargs):
        pass

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
            # if density is None and self.prev_density is None:
                # raise AttributeError("Must provide a guess density")
            # elif density is not None :
                # self.prev_density = density

            if gsystem is None and self.evaluator.gsystem is None :
                raise AttributeError("Must provide global system")
            else:
                self.evaluator.gsystem = gsystem

            # rho_ini = self.prev_density.copy()
            # self.density = self.get_density(rho_ini, **kwargs)
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
