from ..evaluator import EnergyEvaluatorMix


class OptDriver:

    def __init__(self, embed_evaluator = None, sub_evaluator = None, options=None, calculator = None, **kwargs):
        default_options = {
        }
        self.options = default_options
        if options is not None :
            self.options.update(options)

        # if sub_evaluator is None:
            # raise AttributeError("Must provide an subsystem functionals")
        # else:
            # self.sub_evaluator = sub_evaluator

        self.energy_evaluator = None
        self.get_energy_evaluator(embed_evaluator, sub_evaluator, **kwargs)
        self.calculator = calculator
        if self.calculator is not None :
            self.calculator.evaluator = self.energy_evaluator

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
        if density is None and self.rho is None:
            raise AttributeError("Must provide a guess density")
        elif density is not None :
            self.rho = density

        if gsystem is None:
            raise AttributeError("Must provide global system")
        else:
            self.energy_evaluator.gsystem = gsystem

        rho_ini = self.rho
        self.energy_evaluator.rest_rho = gsystem.sub_value(gsystem.density, rho_ini) - rho_ini
        #-----------------------------------------------------------------------
        # self.density = self.calculator.get_density(rho_ini)
        self.calculator.get_density(rho_ini)
        self.density = self.calculator.update_density()
        self.mu = self.calculator.get_fermi_level()

        if 'E' in calcType or 'V' in calcType :
            func = self.get_sub_energy(self.rho, calcType)
            self.functional = func

        if 'E' in calcType :
            self.energy = func.energy

        if 'V' in calcType :
            self.potential = func.potential

        return
