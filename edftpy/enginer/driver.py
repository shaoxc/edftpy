from abc import ABC, abstractmethod
from edftpy.utils.common import Field

class Driver(ABC):
    def __init__(self, options=None, technique = 'KS', **kwargs):
        default_options = {
                'update_delay' : 1,
                'update_freq' : 1
        }
        self.options = default_options
        if options is not None :
            self.options.update(options)
        self.technique = technique

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

    def __call__(self, density = None, gsystem = None, calcType = ['O', 'E'], ext_pot = None, **kwargs):
        return self.compute(density, gsystem, calcType, ext_pot, **kwargs)

    def compute(self, density = None, gsystem = None, calcType = ['O', 'E'], ext_pot = None, **kwargs):
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
