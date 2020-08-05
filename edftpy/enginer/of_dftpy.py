import copy
import numpy as np
from functools import partial

from ..mixer import LinearMixer, PulayMixer
from ..utils.common import AbsDFT
from ..utils.math import grid_map_data
from .hamiltonian import Hamiltonian

from dftpy.optimization import Optimization
from dftpy.formats.io import write
from dftpy.external_potential import ExternalPotential


class DFTpyOF(AbsDFT):
    """description"""
    def __init__(self, evaluator = None, subcell = None, options = None, mixer = None,
            grid = None, evaluator_of = None, **kwargs):
        default_options = {
            "opt_method" : 'full',
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
        self.fermi = None
        self.calc = None
        self._iter = 0
        self.phi = None
        self.residual_norm = 1
        self.subcell = subcell
        #-----------------------------------------------------------------------
        self.mixer = mixer
        if self.mixer is None and self.options['opt_method'] != 'full' :
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0, 1.0], maxm = 7, coef = [0.2], predecut = 0, delay = 1)
        #-----------------------------------------------------------------------
        self.density = self.subcell.density
        self.grid_driver = grid
        self.evaluator_of = evaluator_of

    @property
    def grid(self):
        return self.subcell.grid

    def init_density(self, rho_ini = None):
        pass

    def _format_density(self, density, volume = None, sym = True, **kwargs):
        if self.grid_driver is not None :
            self.charge = grid_map_data(density, grid = self.grid_driver)
        else :
            self.charge = density.copy()

    def _format_density_invert(self, charge = None, grid = None, **kwargs):
        if charge is None :
            charge = self.charge

        if grid is None :
            grid = self.grid

        if self.grid_driver is not None and np.any(self.grid_driver.nr != grid.nr):
            rho = grid_map_data(charge, grid = grid)
        else :
            rho = charge.copy()
        return rho

    def _get_extpot(self, density = None, charge= None, grid = None, with_global = False, first = False, **kwargs):
        self._map_gsystem(charge, grid)
        # rho = self._format_density_invert(charge, grid) # Fine grid
        self.evaluator.get_embed_potential(density, gaussian_density = self.subcell.gaussian_density, with_global = with_global, with_ke = True)
        if self.grid_driver is not None :
            self.evaluator_of.embed_potential = grid_map_data(self.evaluator.embed_potential, grid = self.grid_driver)
        else :
            self.evaluator_of.embed_potential = self.evaluator.embed_potential
        return

    def _map_gsystem(self, charge = None, grid = None):
        if self.evaluator_of.gsystem is None :
            self.evaluator_of.gsystem = self.evaluator.gsystem
        elif self.grid_driver is not None :
            self.evaluator_of.gsystem.density = grid_map_data(self.evaluator.gsystem.density, grid = self.evaluator_of.gsystem.grid)
        elif self.evaluator_of.gsystem is self.evaluator.gsystem :
            pass
        else :
            self.evaluator_of.gsystem.density = self.evaluator.gsystem.density
        #-----------------------------------------------------------------------
        self.evaluator_of.rest_rho = self.evaluator_of.gsystem.sub_value(self.evaluator_of.gsystem.density, charge) - charge 
        # self.evaluator_of.rest_rho = self.evaluator.rest_rho
        return

    def get_density(self, density, res_max = None, **kwargs):
        self._iter += 1
        #-----------------------------------------------------------------------
        if res_max is None :
            norm = self.residual_norm
        else :
            norm = res_max

        if self._iter == 1 :
            self.options['econv0'] = self.options['econv'] * 1E4
            self.options['econv'] = self.options['econv0']
            self.residual_norm = 1
        elif self._iter < 3 :
            norm = max(0.1, self.residual_norm)

        econv = self.options['econv0'] * norm
        if econv < self.options['econv'] :
            self.options['econv'] = econv
        if norm < 1E-8 :
            self.options['maxiter'] = 4
        print('econv', self.options['econv'])
        #-----------------------------------------------------------------------
        if self.options['opt_method'] == 'full' :
            hamil = False
        else :
            hamil = True
        if hamil or self._iter == 1 :
            self._format_density(density)
        # self._format_density(density)
        self._get_extpot(density, self.charge, density.grid, with_global = hamil)
        if self.evaluator_of.sub_evaluator and hamil:
            self.evaluator_of.embed_potential += self.evaluator_of.sub_evaluator(self.charge, calcType = ['V']).potential
        self.prev_charge = self.charge.copy()
        #-----------------------------------------------------------------------
        if self.options['opt_method'] == 'full' :
            self.get_density_full_opt(density, **kwargs)
        elif self.options['opt_method'] == 'part' :
            self.get_density_embed(density, **kwargs)
        elif self.options['opt_method'] == 'hamiltonian' :
            self.get_density_hamiltonian(density)
        #-----------------------------------------------------------------------
        rho = self._format_density_invert(self.charge, density.grid)
        return rho

    def get_density_full_opt(self, density, **kwargs):
        remove_global = {}
        embed_evaluator = self.evaluator.embed_evaluator
        total_evaluator = self.evaluator_of.gsystem.total_evaluator
        # if self.evaluator_of.gsystem is self.evaluator.gsystem :
        if embed_evaluator is not None :
            keys_emb = embed_evaluator.funcdicts.keys()
            keys_global = total_evaluator.funcdicts.keys()
            keys = [key for key in keys_global if key in keys_emb]
            for key in keys :
                remove_global[key] = total_evaluator.funcdicts[key]
        total_evaluator.update_functional(remove = remove_global)
        #-----------------------------------------------------------------------
        if 'method' in self.options :
            optimization_method = self.options['method']
        else :
            optimization_method = 'CG-HS'

        evaluator = partial(self.evaluator_of.compute, calcType=["E","V"], with_global = True, with_ke = True, with_embed = True)
        self.calc = Optimization(EnergyEvaluator=evaluator, guess_rho=self.charge, optimization_options=self.options, optimization_method = optimization_method)
        self.calc.optimize_rho()

        self.charge = self.calc.rho
        self.fermi_level = self.calc.mu
        total_evaluator.update_functional(add = remove_global)
        return

    def get_density_embed(self, density, **kwargs):
        if 'method' in self.options :
            optimization_method = self.options['method']
        else :
            optimization_method = 'CG-HS'

        evaluator = partial(self.evaluator_of.compute, with_global = False, with_embed = True, with_ke = True, with_sub = False)
        self.calc = Optimization(EnergyEvaluator=evaluator, guess_rho=density, optimization_options=self.options, optimization_method = optimization_method)
        self.calc.optimize_rho()

        # self.phi = self.calc.phi.copy()
        self.charge = self.calc.rho
        self.fermi_level = self.calc.mu
        return 

    def get_density_hamiltonian(self, density, **kwargs):
        potential = self.evaluator_of.embed_potential
        hamiltonian = Hamiltonian(potential, grid = self.subcell.grid)
        eigens = hamiltonian.eigens()
        eig = eigens[0][0]
        print('eig', eig)
        print('min wave', np.min(eigens[0][1]))
        rho = eigens[0][1] ** 2
        self.fermi_level = eig
        self.charge = rho * np.sum(density) / np.sum(rho)
        return 

    def get_kinetic_energy(self, **kwargs):
        pass

    def get_energy(self, density = None, **kwargs):
        if density is None :
            density = self.density
        obj = self.evaluator_of.compute(density, calcType = ['E'], with_sub = True, with_global = False, with_embed = False, with_ke = False)
        return obj.energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], olevel = 1, **kwargs):
        if olevel == 0 :
            func = self.evaluator.compute(density, calcType = calcType, with_global = False)
        elif olevel == 1 :
            func = self.evaluator.compute(density, calcType = calcType, with_global = False, with_ke = False)
        elif olevel == 2 :
            func = self.evaluator.compute(density, calcType = calcType, with_global = False, with_ke = False, with_embed = True)
        func_driver = self.evaluator_of.compute(self.charge, calcType = ['E'], with_sub = True, with_global = False, with_embed = False, with_ke = True)
        func.energy += func_driver.energy
        print('sub_energy_of', func.energy)
        return func

    def update_density(self, **kwargs):
        r = self.charge - self.prev_charge
        self.residual_norm = np.sqrt(np.sum(r * r)/np.size(r))
        print('res_norm_of', self._iter, np.max(abs(r)), np.sqrt(np.sum(r * r)/np.size(r)))
        if self.options['opt_method'] == 'full' :
            rho = self.charge
        else :
            rho = self.mixer(self.prev_charge, self.charge, **kwargs)
        if self.grid_driver is not None :
            rho = grid_map_data(rho, grid = self.grid)
        return rho

    def get_fermi_level(self, **kwargs):
        results = self.fermi_level
        return results

    def get_forces(self, **kwargs):
        forces = np.zeros((self.subcell.ions.nat, 3))
        return forces

    def get_stress(self, **kwargs):
        pass
