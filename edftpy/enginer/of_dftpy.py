import copy
import numpy as np
from functools import partial

from ..mixer import LinearMixer, PulayMixer
from ..utils.common import AbsDFT
from ..utils.math import grid_map_data
from .hamiltonian import Hamiltonian

from dftpy.optimization import Optimization
from dftpy.formats.io import write


class DFTpyOF(AbsDFT):
    """description"""
    def __init__(self, evaluator = None, subcell = None, options = None, mixer = None, **kwargs):
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
        self.prev_density = None
        self.calc = None
        self._iter = 0
        self.phi = None
        self.residual_norm = 1
        self.subcell = subcell
        #-----------------------------------------------------------------------
        self.mixer = mixer
        if self.mixer is None :
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0, 1.0], maxm = 7, coef = [0.2], predecut = 0, delay = 1)
            # self.mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.2], maxm = 7, coef = [0.2], predecut = 0, delay = 1)
        #-----------------------------------------------------------------------
        self.density = self.subcell.density

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
        elif self._iter < 2 :
            norm = max(0.1, self.residual_norm)

        econv = self.options['econv0'] * norm
        if econv < self.options['econv'] :
            self.options['econv'] = econv
            # if self.options['econv'] < self.options['econv0'] / 1E4 :
                # self.options['econv'] = self.options['econv0'] / 1E4
            # if self.options['econv'] < 1E-12 * self.subcell.ions.nat : self.options['econv'] = 1E-12 * self.subcell.ions.nat
        if norm < 1E-8 :
            self.options['maxiter'] = 4
        #-----------------------------------------------------------------------
        print('econv', self.options['econv'])
        self.prev_density = density.copy()
        if self.options['opt_method'] == 'part' :
            results = self.get_density_embed(density, **kwargs)
        elif self.options['opt_method'] == 'full' :
            results = self.get_density_full_opt(density, **kwargs)
        elif self.options['opt_method'] == 'hamiltonian' :
            results = self.get_density_hamiltonian(density)
        return results

    def get_density_full_opt(self, density, **kwargs):
        self.evaluator.get_embed_potential(density, gaussian_density = self.subcell.gaussian_density, with_global = False, with_sub = False, with_ke = True)
        if 'method' in self.options :
            optimization_method = self.options['method']
        else :
            optimization_method = 'CG-HS'

        if self.evaluator.grid_coarse is None :
            rho = density
            evaluator = self.evaluator.compute_embed
            self.calc = Optimization(EnergyEvaluator=evaluator, guess_rho=rho, optimization_options=self.options, optimization_method = optimization_method)
            self.calc.optimize_rho()
            self.density = self.calc.rho
            # self.density.grid = density.grid
        else :
            rho = grid_map_data(density, grid = self.evaluator.grid_coarse)
            evaluator = self.evaluator.compute_embed_coarse
            self.calc = Optimization(EnergyEvaluator=evaluator, guess_rho=rho, optimization_options=self.options, optimization_method = optimization_method)
            self.calc.optimize_rho(guess_phi = self.phi)
            self.density = grid_map_data(self.calc.rho, grid = density.grid)
            self.density *= np.sum(density)/np.sum(self.density)
            self.phi = self.calc.phi.copy()
        self.fermi_level = self.calc.mu
        return self.density

    def get_density_embed(self, density, lphi = False, **kwargs):

        if 'method' in self.options :
            optimization_method = self.options['method']
        else :
            optimization_method = 'CG-HS'
        # if self._iter > 20 :
            # self.options['econv'] = self.options['econv0'] * self.residual_norm /1E2
            # # self.options['econv'] = self.options['econv0'] * self.residual_norm /1E4
            # if self.options['econv'] < 1E-14 * self.subcell.ions.nat : self.options['econv'] = 1E-14 * self.subcell.ions.nat
        self.evaluator.get_embed_potential(density, gaussian_density = self.subcell.gaussian_density, with_global = True)
        # # evaluator = partial(self.evaluator.compute, with_global = False)
        evaluator = self.evaluator.compute_only_ke
        self.calc = Optimization(EnergyEvaluator=evaluator, guess_rho=density, optimization_options=self.options, optimization_method = optimization_method)
        self.calc.optimize_rho(guess_rho = density)

        # self.calc.optimize_rho(guess_rho = density, lphi = True)
        # self.calc.optimize_rho(guess_rho = density, guess_phi = self.phi, lphi = lphi)
        # self.calc.optimize_rho(guess_rho = density, guess_phi = self.phi, lphi = True)
        # if self.calc.converged > 0 :
            # print('Saved phi make DFTpy not converged, try turn off lphi')
            # self.calc.optimize_rho(guess_rho = density, guess_phi = self.phi, lphi = False)
        # if self.calc.converged > 0 :
            # print('Saved phi make DFTpy not converged, try new phi')
            # self.calc.optimize_rho(guess_rho = density, lphi = lphi)
        # if self.calc.converged > 0 :
            # print('Saved phi make DFTpy not converged, try new all')
            # self.calc.optimize_rho(guess_rho = density)
        # if self.calc.converged == 1 :
            # print("!WARN: DFTpy not converged")
            # raise AttributeError("!!!ERROR : DFTpy not converged")
        self.density = self.calc.rho
        self.fermi_level = self.calc.mu
        self.phi = self.calc.phi.copy()
        return self.density

    def get_density_hamiltonian(self, density, **kwargs):
        self.evaluator.get_embed_potential(density, gaussian_density = self.subcell.gaussian_density, with_global = True)
        # potential = self.evaluator(density).potential
        potential = self.evaluator.embed_potential
        hamiltonian = Hamiltonian(potential, grid = self.subcell.grid)
        eigens = hamiltonian.eigens()
        eig = eigens[0][0]
        print('eig', eig)
        print('min wave', np.min(eigens[0][1]))
        rho = eigens[0][1] ** 2
        self.density = rho * np.sum(density) / np.sum(rho)
        self.fermi_level = eig
        return self.density

    def get_kinetic_energy(self, **kwargs):
        pass

    def get_energy(self, density = None, **kwargs):
        if density is None :
            density = self.density
        energy = self.evaluator(density, calcType = ['E'], with_global = False, embed = False).energy
        # energy = self.evaluator(density, calcType = ['E'], with_global = False, embed = False, only_ke = True).energy
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], **kwargs):
        if self.options['opt_method'] == 'full' :
            # func = self.evaluator(density, calcType = calcType, with_global = False, embed = False)
            func = self.evaluator.compute(density, calcType = calcType, with_global = False)
        else :
            # func = self.evaluator(density, calcType = ['E'], with_global = False, embed = False, only_ke = True)
            func = self.evaluator.compute_only_ke(density, calcType = calcType, with_global = False, embed = False)
        print('sub_energy_of', func.energy)
        return func

    def update_density(self, **kwargs):
        r = self.density - self.prev_density
        self.residual_norm = np.sqrt(np.sum(r * r)/np.size(r))
        print('res_norm_of', self._iter, np.max(abs(r)), np.sqrt(np.sum(r * r)/np.size(r)))
        self.density = self.mixer(self.prev_density, self.density, **kwargs)
        # print('max',self.density.max(), self.density.min(), self.density.integral())
        # write(str(self._iter) + '.xsf', self.density, self.subcell.ions)
        return self.density

    def get_fermi_level(self, **kwargs):
        results = self.fermi_level
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
