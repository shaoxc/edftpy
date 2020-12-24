import copy
import numpy as np
from functools import partial
import sys

from ..mixer import LinearMixer, PulayMixer, AbstractMixer
from ..utils.math import grid_map_data
from .hamiltonian import Hamiltonian
from .driver import Driver
from edftpy.mpi import sprint

from dftpy.optimization import Optimization
from dftpy.formats.io import write
from dftpy.external_potential import ExternalPotential
import dftpy.constants


class DFTpyOF(Driver):
    """description"""
    def __init__(self, evaluator = None, subcell = None, prefix = 'sub_of', options = None, mixer = None,
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
        super().__init__(options = self.options, technique = 'OF')

        self.evaluator = evaluator
        self.subcell = subcell
        self.grid_driver = grid
        self.evaluator_of = evaluator_of
        self.prefix = prefix
        #-----------------------------------------------------------------------
        self.mixer = mixer
        if self.options['opt_method'] == 'full' :
            if not isinstance(self.mixer, AbstractMixer):
                self.mixer = LinearMixer(predtype = None, coef = [1.0], predecut = None, delay = 1)
        elif self.mixer is None :
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0, 1.0], maxm = 7, coef = [0.2], predecut = 0, delay = 1)
        elif isinstance(self.mixer, float):
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0, 1.0], maxm = 7, coef = self.mixer, predecut = 0, delay = 1)
        #-----------------------------------------------------------------------
        self.density = self.subcell.density
        self.init_density()
        self.comm = self.subcell.grid.mp.comm
        self.outfile = self.prefix + '.out'
        if self.comm.rank == 0 :
            self.fileobj = open(self.outfile, 'w')
        else :
            self.fileobj = None
        self.update_workspace(first = True)

    def update_workspace(self, subcell = None, first = False, **kwargs):
        """
        Notes:
            clean workspace
        """
        self.rho = None
        self.wfs = None
        self.occupations = None
        self.eigs = None
        self.fermi = None
        self.calc = None
        self._iter = 0
        self.energy = 0.0
        self.phi = None
        self.residual_norm = 1
        if isinstance(self.mixer, AbstractMixer):
            self.mixer.restart()
        if subcell is not None :
            self.subcell = subcell
        self.gaussian_density = self.get_gaussian_density(self.subcell, grid = self.grid)
        if not first :
            self.options['econv'] = self.options['econv0'] / 1E4
        return

    @property
    def grid(self):
        return self.subcell.grid

    def init_density(self, rho_ini = None):
        self.prev_density = self.density.copy()
        if self.grid_driver is not None :
            self.charge = grid_map_data(self.density, grid = self.grid_driver)
            self.prev_charge = self.charge.copy()
        else :
            self.charge = self.density
            self.prev_charge = self.prev_density

    def _format_density(self, sym = True, **kwargs):
        self.prev_density[:] = self.density
        if self.grid_driver is not None :
            self.prev_charge, self.charge = self.charge, self.prev_charge
            self.charge[:] = grid_map_data(self.density, grid = self.grid_driver)
        else :
            self.charge = self.density
            self.prev_charge = self.prev_density

    def _format_density_invert(self, charge = None, grid = None, **kwargs):
        if charge is None :
            charge = self.charge

        if grid is None :
            grid = self.grid

        if self.grid_driver is not None and np.any(self.grid_driver.nr != grid.nr):
            self.density[:]= grid_map_data(charge, grid = grid)
        return self.density

    def _get_extpot(self, with_global = False, first = False, **kwargs):
        self._map_gsystem()
        gsystem = self.evaluator.gsystem
        embed_keys = []
        if self.evaluator.embed_evaluator is not None :
            embed_keys = self.evaluator.embed_evaluator.funcdicts.keys()
        gsystem.total_evaluator.get_embed_potential(gsystem.density, gaussian_density = gsystem.gaussian_density, embed_keys = embed_keys, with_global = with_global)
        self.evaluator.get_embed_potential(self.density, gaussian_density = self.subcell.gaussian_density, with_ke = True)
        gsystem.add_to_sub(gsystem.total_evaluator.embed_potential, self.evaluator.embed_potential)
        if self.grid_driver is not None :
            self.evaluator_of.embed_potential = grid_map_data(self.evaluator.embed_potential, grid = self.grid_driver)
        else :
            self.evaluator_of.embed_potential = self.evaluator.embed_potential
        return

    def _map_gsystem(self):
        if self.evaluator_of.gsystem is None :
            self.evaluator_of.gsystem = self.evaluator.gsystem
        elif self.grid_driver is not None :
            self.evaluator_of.gsystem.density = grid_map_data(self.evaluator.gsystem.density, grid = self.evaluator_of.gsystem.grid)
        elif self.evaluator_of.gsystem is self.evaluator.gsystem :
            pass
        else :
            self.evaluator_of.gsystem.density = self.evaluator.gsystem.density
        #-----------------------------------------------------------------------
        self.evaluator_of.set_rest_rho(self.charge)
        return

    def get_density(self, res_max = None, **kwargs):
        self._iter += 1
        #-----------------------------------------------------------------------
        if res_max is None :
            res_max = self.residual_norm

        if self._iter == 1 :
            self.options['econv0'] = self.options['econv'] * 1E4
            self.options['econv'] = self.options['econv0']
            res_max = 1.0

        norm = res_max
        if self._iter < 3 :
            norm = max(0.1, res_max)

        econv = self.options['econv0'] * norm
        # if econv / self.options['econv'] < 1E-2 :
            # econv = self.options['econv'] * 1E-2
        if econv < self.options['econv'] :
            self.options['econv'] = econv
        if norm < 1E-7 :
            self.options['maxiter'] = 4
        sprint('econv', self._iter, self.options['econv'], comm=self.comm)
        #-----------------------------------------------------------------------
        if self.options['opt_method'] == 'full' :
            hamil = False
        else :
            hamil = True
        if hamil or self._iter == 1 :
            self._format_density()

        self._get_extpot(with_global = hamil)
        if self.evaluator_of.sub_evaluator and hamil:
            self.evaluator_of.embed_potential += self.evaluator_of.sub_evaluator(self.charge, calcType = ['V']).potential

        self.prev_charge[:] = self.charge
        #-----------------------------------------------------------------------
        stdout = dftpy.constants.STDOUT
        dftpy.constants.STDOUT = self.fileobj
        if self.options['opt_method'] == 'full' :
            self.get_density_full_opt(**kwargs)
        elif self.options['opt_method'] == 'part' :
            self.get_density_embed(**kwargs)
        elif self.options['opt_method'] == 'hamiltonian' :
            self.get_density_hamiltonian(**kwargs)
        dftpy.constants.STDOUT = stdout
        #-----------------------------------------------------------------------
        self._format_density_invert(self.charge, self.grid)
        return self.density

    def get_density_full_opt(self, **kwargs):
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

        self.charge[:] = self.calc.rho
        self.fermi_level = self.calc.mu
        total_evaluator.update_functional(add = remove_global)
        return

    def get_density_embed(self, **kwargs):
        if 'method' in self.options :
            optimization_method = self.options['method']
        else :
            optimization_method = 'CG-HS'

        evaluator = partial(self.evaluator_of.compute, with_global = False, with_embed = True, with_ke = True, with_sub = False)
        self.calc = Optimization(EnergyEvaluator=evaluator, guess_rho=self.charge, optimization_options=self.options, optimization_method = optimization_method)
        self.calc.optimize_rho()

        # self.phi = self.calc.phi.copy()
        self.charge[:] = self.calc.rho
        self.fermi_level = self.calc.mu
        return

    def get_density_hamiltonian(self, num_eig = 2, **kwargs):
        potential = self.evaluator_of.embed_potential
        hamiltonian = Hamiltonian(potential, grid = self.subcell.grid)
        self.options.update(kwargs)
        if num_eig > 1 :
            self.options['eig_tol'] = 1E-6
        eigens = hamiltonian.eigens(num_eig, **self.options)
        sprint('eigens', ' '.join(map(str, [ev[0] for ev in eigens])), comm=self.comm)
        eig = eigens[0][0]
        sprint('eig', eig, comm=self.comm)
        sprint('min wave', np.min(eigens[0][1]), comm=self.comm)
        rho = eigens[0][1] ** 2
        self.fermi_level = eig
        self.charge[:] = rho * self.charge.integral() / rho.integral()
        return eigens

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
        sprint('sub_energy_of', func.energy, comm=self.comm)
        func.energy /= self.grid.mp.size
        return func

    def update_density(self, **kwargs):
        r = self.charge - self.prev_charge
        self.residual_norm = np.sqrt(self.grid.mp.asum(r * r)/self.grid.nnrR)
        rmax = r.amax()
        sprint('res_norm_of', self._iter, rmax, self.residual_norm, comm=self.comm)
        if self.options['opt_method'] == 'full' :
            rho = self.charge
        else :
            rho = self.mixer(self.prev_charge, self.charge, **kwargs)
        self.charge[:] = rho
        self._format_density_invert(self.charge, self.grid)
        return self.density

    def get_fermi_level(self, **kwargs):
        results = self.fermi_level
        return results

    def get_forces(self, **kwargs):
        forces = np.zeros((self.subcell.ions.nat, 3))
        return forces

    def get_stress(self, **kwargs):
        pass

def dftpy_opt(ions, rho, pplist, xc_kwargs = None, ke_kwargs = None, stdout = None):
    from dftpy.interface import OptimizeDensityConf
    from dftpy.config import DefaultOption, OptionFormat
    from dftpy.system import System

    from edftpy.pseudopotential import LocalPP
    from edftpy.kedf import KEDF
    from edftpy.hartree import Hartree
    from edftpy.xc import XC
    from edftpy.evaluator import Evaluator
    #-----------------------------------------------------------------------
    save = dftpy.constants.STDOUT
    if stdout is None :
        dftpy.constants.STDOUT = stdout
    if xc_kwargs is None :
        xc_kwargs = {"x_str":'gga_x_pbe','c_str':'gga_c_pbe'}
    if ke_kwargs is None :
        ke_kwargs = {'name' :'GGA', 'k_str' : 'REVAPBEK'}
        # ke_kwargs = {'name' :'TFvW', 'y' :0.2}
    #-----------------------------------------------------------------------
    pseudo = LocalPP(grid = rho.grid, ions=ions, PP_list=pplist, PME=True)
    hartree = Hartree()
    xc = XC(**xc_kwargs)
    ke = KEDF(**ke_kwargs)
    funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
    evaluator = Evaluator(**funcdicts)
    #-----------------------------------------------------------------------
    conf = DefaultOption()
    conf['JOB']['calctype'] = 'Density'
    conf['OUTPUT']['time'] = False
    conf['OPT'] = {"method" :'CG-HS', "maxiter": 1000, "econv": 1.0e-6*ions.nat}
    conf = OptionFormat(conf)
    struct = System(ions, rho.grid, name='density', field=rho)
    opt = OptimizeDensityConf(conf, struct, evaluator)
    dftpy.constants.STDOUT = save
    # rho[:] =opt['density'] 
    # return rho
    return opt['density']
