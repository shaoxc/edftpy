from ..evaluator import EnergyEvaluatorMix
from ..hartree import Hartree


class OptDriver:

    def __init__(self, energy_evaluator = None, embed_evaluator = None, sub_evaluator = None, options=None, calculator = None, core_density = None, **kwargs):
        default_options = {
                'update_delay' : 5, 
                'update_freq' : 1
        }
        self.options = default_options
        if options is not None :
            self.options.update(options)

        self.energy_evaluator = energy_evaluator
        self.calculator = calculator
        self.core_density = core_density
        if self.energy_evaluator is None :
            self.get_energy_evaluator(embed_evaluator, sub_evaluator, **kwargs)
        if self.calculator is not None :
            if self.calculator.evaluator is None :
                self.calculator.evaluator = self.energy_evaluator
            else :
                self.energy_evaluator = self.calculator.evaluator

        self.energy_traj = {
                'KE': [], 
                'XC': [], 
                'HARTREE': []}
        self.prev_density = None
        self._it = 0

    def get_energy_evaluator(self, embed_evaluator = None, sub_evaluator = None, **kwargs):
        """
        """
        if self.energy_evaluator is None :
            self.energy_evaluator = EnergyEvaluatorMix(embed_evaluator, sub_evaluator, **kwargs)
        return self.energy_evaluator

    def get_sub_energy(self, density, calcType=["E","V"]):
        obj = self.calculator.get_energy_potential(density, calcType)
        return obj

    def __call__(self, density = None, gsystem = None, calcType = ['O', 'E'], ext_pot = None):
        return self.compute(density, gsystem, calcType, ext_pot)

    def compute(self, density = None, gsystem = None, calcType = ['O', 'E'], ext_pot = None):
        #-----------------------------------------------------------------------
        self._it += 1
        # update_delay = self.options['update_delay']
        # update_freq = self.options['update_freq']
        # if self._it > update_delay and (self._it - update_delay) % update_freq > 0:
            # return
        #-----------------------------------------------------------------------
        if density is None and self.prev_density is None:
            raise AttributeError("Must provide a guess density")
        elif density is not None :
            self.prev_density = density

        if gsystem is None:
            raise AttributeError("Must provide global system")
        else:
            self.energy_evaluator.gsystem = gsystem

        rho_ini = self.prev_density
        #-----------------------------------------------------------------------
        self.energy_evaluator.rest_rho = gsystem.sub_value(gsystem.density, rho_ini) - rho_ini
        self.energy_evaluator.get_embed_potential(rho_ini, core_density = self.core_density)
        #-----------------------------------------------------------------------
        self.density = self.calculator.get_density(rho_ini)
        # self.calculator.get_density(rho_ini)
        # self.density = self.calculator.update_density()
        self.mu = self.calculator.get_fermi_level()

        if 'E' in calcType or 'V' in calcType :
            func = self.get_sub_energy(self.prev_density, calcType)
            self.functional = func

        if 'E' in calcType :
            self.energy = func.energy

        if 'V' in calcType :
            self.potential = func.potential

        return

    def update_density(self, **kwargs):
        return self.calculator.update_density(**kwargs)

    def get_energy_traj(self, ename = 'HARTREE', density = None):
        # if ename is not None :
            # energy = self.calculator.get_energy_part(ename = ename, density = density)
        if ename == 'HARTREE' :
            energy = Hartree.compute(density, calcType=['E']).energy
            print('Hartree energy', energy)
            self.energy_traj[ename].append(energy)
        else :
            raise AttributeError("!ERROR : Not implemented", ename)

        return self.energy_traj[ename]

