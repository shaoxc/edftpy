import copy
import numpy as np
from functools import partial

from ..mixer import LinearMixer, PulayMixer
from ..utils.common import AbsDFT
from .hamiltonian import Hamiltonian

from dftpy.optimization import Optimization
from dftpy.formats.io import write


class DFTpyOF(AbsDFT):
    """description"""
    def __init__(self, evaluator = None, grid = None, rho_ini = None, options = None, mixer = None, 
            ions = None, gaussian_density = None, **kwargs):
        default_options = {
            "opt_method" : 'part',
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
        self.gaussian_density = gaussian_density
        self._grid = grid
        self._ions = ions
        self.rho = None
        self.fermi = None
        self.density = None
        self.prev_density = None
        self.calc = None
        self._iter = 0
        self.phi = None
        self.residual_norm = 1
        #-----------------------------------------------------------------------
        self.mixer = mixer
        if self.mixer is None :
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0, 1.0], maxm = 7, coef = [0.2], predecut = 0, delay = 1)
            # self.mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.2], maxm = 7, coef = [0.2], predecut = 0, delay = 1)
            # self.mixer = LinearMixer(predtype = 'kerker', coef = [0.7], predecut = 0, delay = 1)
        #-----------------------------------------------------------------------

    @property
    def grid(self):
        if self._grid is None :
            if hasattr(self.prev_density, 'grid'):
                self._grid = self.prev_density.grid
            elif hasattr(self.density, 'grid'):
                self._grid = self.density.grid
            else :
                raise AttributeError("Must set grid firstly")
        return self._grid

    def get_density(self, density, **kwargs):
        self._iter += 1
        #-----------------------------------------------------------------------
        if self._iter == 1 :
            self.options['econv0'] = self.options['econv'] * 1E4
            self.residual_norm = 1
            self.options['econv'] = self.options['econv0'] * self.residual_norm
        econv = self.options['econv0'] * self.residual_norm
        if econv < self.options['econv'] :
            self.options['econv'] = econv
            if self._ions is not None :
                if self.options['econv'] < 1E-14 * self._ions.nat : self.options['econv'] = 1E-14 * self._ions.nat
        #-----------------------------------------------------------------------
        self.prev_density = density.copy()
        if self.options['opt_method'] == 'part' :
            results = self.get_density_embed(density, **kwargs)
        elif self.options['opt_method'] == 'full' :
            results = self.get_density_full_opt(density, **kwargs)
        elif self.options['opt_method'] == 'hamiltonian' :
            results = self.get_density_hamiltonian(density)
        # np.savetxt('of_den.' + str(self._iter), np.c_[self.grid.get_reciprocal().q.real.ravel(), self.density.fft().real.ravel()])
        return results

    def get_density_full_opt(self, density, **kwargs):
        #-----------------------------------------------------------------------
        # if self._iter == 1 :
            # self.options['econv0'] = self.options['econv'] * 1E4
            # self.residual_norm = 1
            # self.options['econv'] = self.options['econv0'] * self.residual_norm
        # econv = self.options['econv0'] * self.residual_norm
        # if econv < self.options['econv'] :
            # self.options['econv'] = econv
        #-----------------------------------------------------------------------
        # self.evaluator.get_embed_potential(density, gaussian_density = self.gaussian_density, with_global = False, with_sub = False, with_ke = False)
        self.evaluator.get_embed_potential(density, gaussian_density = self.gaussian_density, with_global = False, with_sub = False, with_ke = True)
        evaluator = self.evaluator.compute_embed
        if 'method' in self.options :
            optimization_method = self.options['method']
        else :
            optimization_method = 'CG-HS'
        self.calc = Optimization(EnergyEvaluator=evaluator, guess_rho=density, optimization_options=self.options, optimization_method = optimization_method)
        self.calc.optimize_rho()
        self.density = self.calc.rho
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
            # if self.options['econv'] < 1E-14 * self._ions.nat : self.options['econv'] = 1E-14 * self._ions.nat
        self.evaluator.get_embed_potential(density, gaussian_density = self.gaussian_density, with_global = True)
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
        self.evaluator.get_embed_potential(density, gaussian_density = self.gaussian_density, with_global = True)
        # potential = self.evaluator(density).potential
        potential = self.evaluator.embed_potential
        hamiltonian = Hamiltonian(potential, grid = self.grid)
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
        # energy = self.evaluator(density, calcType = ['E'], with_global = False, embed = False).energy
        energy = self.evaluator(density, calcType = ['E'], with_global = False, embed = False, only_ke = True).energy
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], **kwargs):
        if self.options['opt_method'] == 'full' :
            func = self.evaluator(density, calcType = calcType, with_global = False, embed = False)
        else :
            func = self.evaluator(density, calcType = ['E'], with_global = False, embed = False, only_ke = True)
        print('sub_energy_of', func.energy)
        return func

    def update_density(self, **kwargs):
        r = self.density - self.prev_density
        self.residual_norm = np.sqrt(np.sum(r * r)/np.size(r))
        print('res_norm_of', self._iter, np.max(abs(r)), np.sqrt(np.sum(r * r)/np.size(r)))
        self.density = self.mixer(self.prev_density, self.density, **kwargs)
        # print('max',self.density.max(), self.density.min(), self.density.integral())
        # write(str(self._iter) + '.xsf', self.density, self._ions)
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
