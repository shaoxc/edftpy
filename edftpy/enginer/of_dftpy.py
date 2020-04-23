import copy
from ..mixer import LinearMixer
from ..utils.common import AbsDFT

from dftpy.optimization import Optimization


class DFTpyOF(AbsDFT):
    """description"""
    def __init__(self, evaluator = None, grid = None, rho_ini = None, options = None, **kwargs):
        default_options = {
            "method" :'CG-HS',
            "maxcor": 6,
            "ftol": 1.0e-7,
            "xtol": 1.0e-12,
            "maxfun": 50,
            "maxiter": 200,
            "maxls": 30,
            "econv": 1.0e-3,
            "c1": 1e-4,
            "c2": 0.2,
            "algorithm": "EMM",
            "vector": "Orthogonalization",
        }
        self.options = default_options
        if options is not None :
            self.options.update(options)

        self.evaluator = evaluator
        self.grid = grid
        self.rho = None
        self.fermi = None
        self.density = None
        self.prev_density = None
        self.mdl = None
        self.mixer = LinearMixer()

    def get_density(self, density, vext = None, **kwargs):
        self.mdl = Optimization(EnergyEvaluator=self.evaluator, guess_rho=density, optimization_options=self.options)
        self.mdl.optimize_rho()
        self.prev_density = copy.deepcopy(density)
        return self.mdl.rho

    def get_kinetic_energy(self, **kwargs):
        pass

    def get_energy(self, density = None, **kwargs):
        if density is None :
            density = self.mdl.rho
        energy = self.evaluator(density, calcType = ['E'], with_global = False).energy
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], **kwargs):
        func = self.evaluator(density, calcType = calcType, with_global = False)
        return func

    def update_density(self, **kwargs):
        self.density = self.mixer(self.prev_density, self.mdl.rho, coef=[0.1])
        return self.density

    def get_fermi_level(self, **kwargs):
        results = self.mdl.mu
        return results
