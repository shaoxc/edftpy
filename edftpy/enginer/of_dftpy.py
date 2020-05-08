import copy
import numpy as np
from ..mixer import LinearMixer, PulayMixer
from ..utils.common import AbsDFT

from dftpy.optimization import Optimization


class DFTpyOF(AbsDFT):
    """description"""
    def __init__(self, evaluator = None, grid = None, rho_ini = None, options = None, mixer = None, **kwargs):
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
        self.calc = None
        self._iter = 0
        #-----------------------------------------------------------------------
        self.mixer = mixer
        if self.mixer is None :
            # self.mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.2], maxm = 7, coef = [1.0], predecut = None, delay = 1)
            # self.mixer = LinearMixer(predtype = None, coef = [0.7], predecut = None, delay = 1)
            self.mixer = LinearMixer(predtype = None, coef = [1.0], predecut = None, delay = 0)
            # self.mixer = LinearMixer(predtype = 'inverse_kerker', predcoef = [0.8, 0.5], maxm = 5, coef = [0.5])
            # self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0], maxm = 7, coef = [1.0], predecut = 20.0, delay = 1)
            # self.mixer = PulayMixer(predtype = None, predcoef = [0.8], maxm = 7, coef = [0.6], predecut = 20.0, delay = 5)
            # self.mixer = BroydenMixer(predtype = 'inverse_kerker', predcoef = [0.2], maxm = 5, coef = [1.0])
            # self.mixer = BroydenMixer(predtype =None, predcoef = [0.2], maxm = 7, coef = [1.0])
        #-----------------------------------------------------------------------

    def get_density(self, density, vext = None, **kwargs):
        self._iter += 1
        if self._iter == 1 :
            self.options['maxiter'] += 60
        elif self._iter == 2 :
            self.options['maxiter'] -= 60

        self.calc = Optimization(EnergyEvaluator=self.evaluator, guess_rho=density, optimization_options=self.options)
        self.calc.optimize_rho()
        self.prev_density = copy.deepcopy(density)
        return self.calc.rho

    def get_kinetic_energy(self, **kwargs):
        pass

    def get_energy(self, density = None, **kwargs):
        if density is None :
            density = self.calc.rho
        energy = self.evaluator(density, calcType = ['E'], with_global = False).energy
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], **kwargs):
        func = self.evaluator(density, calcType = calcType, with_global = False)
        return func

    def update_density(self, **kwargs):
        r = self.calc.rho - self.prev_density
        print('res_norm_of', self._iter, np.max(abs(r)), np.sqrt(np.sum(r * r)/np.size(r)))
        self.density = self.mixer(self.prev_density, self.calc.rho, **kwargs)
        return self.density

    def get_fermi_level(self, **kwargs):
        results = self.calc.mu
        return results

    def get_energy_part(self, ename, density = None, **kwargs):
        evaluator = self.evaluator.gsystem.total_evaluator # Later will replace with sub_evaluator
        # evaluator = self.evaluator.sub_evaluator 
        key = None
        if ename == 'TOTAL' :
            energy = evaluator(density, calcType = ['E'], with_global = False).energy
        elif ename == 'XC' :
            key = 'XC'
        elif ename == 'KEDF' :
            key = 'KEDF'
        elif ename == 'LOCAL' :
            key = 'PSEUDO'
        elif ename == 'HARTREE' :
            key = 'HARTREE'
        elif ename == 'EWALD' :
            raise AttributeError("!ERROR : not contains this energy", ename)
        else :
            raise AttributeError("!ERROR : not contains this energy", ename)
        if key is not None :
            energy = evaluator.funcdicts[key](density, calcType = ['E']).energy
        return energy
