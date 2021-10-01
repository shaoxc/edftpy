import copy
import numpy as np
from functools import partial
import dftpy
from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.optimization import Optimization
from dftpy.constants import environ

from edftpy.mixer import LinearMixer, PulayMixer, AbstractMixer
from edftpy.utils.math import grid_map_data
from edftpy.functional import hartree_energy, KEDFunctional
from edftpy.mpi import sprint
from .hamiltonian import Hamiltonian
from edftpy.engine.engine import Driver

try:
    __version__ = dftpy.__version__
except Exception :
    __version__ = '0.0.1'

class EngineDFTpy(Driver):
    """description"""
    def __init__(self, grid = None, evaluator_of = None, **kwargs):
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
            "interspace": False,
        }
        options = kwargs.get("options", None)
        if options is not None :
            default_options.update(kwargs["options"])
        kwargs["options"] = default_options
        kwargs["technique"] = 'OF'
        super().__init__(**kwargs)
        #-----------------------------------------------------------------------
        self.grid_driver = grid
        self.evaluator_of = evaluator_of
        #-----------------------------------------------------------------------
        if self.mixer is None :
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0, 1.0], maxm = 7, coef = 0.2, predecut = 0, delay = 1)
        # elif isinstance(self.mixer, float):
        #     self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0, 1.0], maxm = 7, coef = self.mixer, predecut = 0, delay = 1)
        if not isinstance(self.mixer, AbstractMixer):
            self.mixer = LinearMixer(predtype = None, coef = 1.0, predecut = None, delay = 1)
        #-----------------------------------------------------------------------
        self.density = self.subcell.density
        self.init_density()
        self.outfile = self.prefix + '.out'
        if self.comm.rank == 0 :
            if self.append :
                self.fileobj = open(self.outfile, 'a')
            else :
                self.fileobj = open(self.outfile, 'w')
        else :
            self.fileobj = None

        fstr = f'Subcell grid({self.prefix}): {self.subcell.grid.nrR}  {self.subcell.grid.nr}\n'
        fstr += f'Subcell shift({self.prefix}): {self.subcell.grid.shift}\n'
        if self.comm.rank == 0 : self.fileobj.write(fstr)
        sprint(fstr, comm=self.comm)

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
        self.dp_norm = 1
        self.core_density = None
        if isinstance(self.mixer, AbstractMixer):
            self.mixer.restart()
        if subcell is not None :
            self.subcell = subcell

        self.gaussian_density = self.get_gaussian_density(self.subcell, grid = self.grid)

        if not first :
            self.options['econv'] = self.options['econv0'] / 1E4

        if self.ncharge is not None and self.ncharge > 0.0:
            nelec = self.density.integral()
            # sprint('nelec', nelec, self.ncharge, self.exttype)
            if self.exttype < 0 :
                self.density[:] = (nelec - self.ncharge) / self.grid.volume
                if self.core_density is not None :
                    self.core_density[:] = 0.0

                if self.gaussian_density is not None :
                    self.gaussian_density_inter = self.gaussian_density.copy()
                    self.gaussian_density[:] = 0.0
                else :
                    self.gaussian_density_inter = None
            else :
                self.density[:] *= (nelec - self.ncharge)/ nelec
            self._format_density()
            sprint('ncharge_sub', self.density.integral())
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

    def get_extpot(self, extpot = None, with_global = True, mapping = True, with_all = False, **kwargs):
        '''
        This is different compare to KS-driver, because here can use different embedding from KS-driver
        '''
        if extpot is None :
            self.evaluator.get_embed_potential(self.density, gaussian_density = self.subcell.gaussian_density, with_ke = True, with_global = False)
            extpot = self.evaluator.embed_potential

        if with_global :
            self._map_gsystem()
            gsystem = self.evaluator.gsystem
            embed_keys = []
            if self.evaluator.nfunc > 0 :
                embed_keys = self.evaluator.funcdicts.keys()
            gsystem.total_evaluator.get_embed_potential(gsystem.density, gaussian_density = gsystem.gaussian_density, embed_keys = embed_keys, with_global = with_all)
            gsystem.add_to_sub(gsystem.total_evaluator.embed_potential, extpot)

        if mapping :
            if self.grid_driver is not None :
                extpot = grid_map_data(extpot, grid = self.grid_driver)
            self.evaluator_of.embed_potential = extpot
        return extpot

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

    def get_density(self, res_max = None, sdft = 'sdft', **kwargs):
        self._iter += 1
        #-----------------------------------------------------------------------
        if res_max is None :
            res_max = self.residual_norm

        if self.comm.size > 1 :
            #only rank0 has the correct residual_norm
            res_max = self.comm.bcast(res_max, root = 0)

        if self._iter == 1 :
            self.options['econv0'] = self.options['econv'] * 1E4
            self.options['econv'] = self.options['econv0']
            res_max = 1.0

        norm = res_max
        if self._iter < 3 :
            norm = max(0.1, res_max)

        econv = self.options['econv0'] * norm
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

        if sdft == 'pdft' :
            if not hamil :
                raise AttributeError("Only 'part' and 'hamiltonian' method can use PDFT")
            extpot = self.evaluator.embed_potential
            #-----------------------------------------------------------------------
            if self.exttype < 0 and self.gaussian_density_inter is not None:
                kepot = KEDFunctional(self.gaussian_density_inter, name = 'GGA', calcType = ['V'], k_str = 'REVAPBEK').potential
                extpot += kepot
                # extpot[self.gaussian_density_inter > 1E-5] = 0.0
            #-----------------------------------------------------------------------
            extpot = self.get_extpot(extpot, mapping = True, with_global = False)
        else :
            self.get_extpot(with_all = hamil)

        if self.evaluator_of.nfunc > 0 and hamil:
            self.evaluator_of.embed_potential += self.evaluator_of(self.charge, calcType = ['V'], with_global = False, with_ke = False, with_embed = False).potential

        self.prev_charge[:] = self.charge
        #-----------------------------------------------------------------------
        stdout = environ['STDOUT']
        environ['STDOUT'] = self.fileobj
        if self.options['opt_method'] == 'full' :
            self.get_density_full_opt(**kwargs)
        elif self.options['opt_method'] == 'part' :
            self.get_density_embed(**kwargs)
        elif self.options['opt_method'] == 'hamiltonian' :
            if self.comm.size > 1 :
                raise AttributeError("Not support parallel")
            self.get_density_hamiltonian(**kwargs)
        environ['STDOUT'] = stdout
        #-----------------------------------------------------------------------
        self._format_density_invert(self.charge, self.grid)
        return self.density

    def get_density_full_opt(self, **kwargs):
        remove_global = {}
        total_evaluator = self.evaluator_of.gsystem.total_evaluator
        if self.evaluator.nfunc > 0 :
            keys_emb = self.evaluator.funcdicts.keys()
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

        evaluator = partial(self.evaluator_of.compute, calcType=["E","V"], with_global = True, with_ke = True, with_embed = True, gather = True)
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

        evaluator = partial(self.evaluator_of.compute, with_global = False, with_embed = True, with_ke = True, with_sub = False, gather = True)
        self.calc = Optimization(EnergyEvaluator=evaluator, guess_rho=self.charge, optimization_options=self.options, optimization_method = optimization_method)
        self.calc.optimize_rho()

        # self.phi = self.calc.phi.copy()
        self.charge[:] = self.calc.rho
        self.fermi_level = self.calc.mu
        return

    def get_density_hamiltonian(self, num_eig = 1, **kwargs):
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

    def get_energy_potential(self, density, calcType = ['E', 'V'], olevel = 1, sdft = 'sdft', **kwargs):
        if olevel == 0 :
            edict = self.evaluator.compute(density, calcType = calcType, with_global = False, split = True)
            func = edict.pop('TOTAL')
        elif olevel == 1 :
            func = self.evaluator.compute(density, calcType = calcType, with_global = False, with_ke = False)
        else : # elif olevel == 2 :
            func = self.evaluator.compute(density, calcType = calcType, with_global = False, with_ke = False, with_embed = True)
        func_driver = self.evaluator_of.compute(self.charge, calcType = ['E'], with_sub = True, with_global = False, with_embed = False, with_ke = True)
        if 'E' in calcType :
            if olevel == 0 :
                fstr = format("Energy information", "-^80") + '\n'
                for key, item in edict.items():
                    energy = self.grid.mp.asum(item.energy)
                    fstr += "{:>12s} energy: {:22.15E} (eV) = {:22.15E} (a.u.)\n".format(key, energy* ENERGY_CONV["Hartree"]["eV"], item.energy)
                fstr += "-" * 80 + '\n'
            else :
                fstr = ''
            func.energy += func_driver.energy
            fstr += f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm, level=1)
            if self.fileobj :
                self.fileobj.write(fstr + '\n')
        return func

    def update_density(self, **kwargs):
        r = self.charge - self.prev_charge
        self.dp_norm = hartree_energy(r)
        self.residual_norm = np.sqrt(self.grid.mp.asum(r * r)/self.grid.nnrR)
        rmax = r.amax()
        fstr = f'res_norm({self.prefix}): {self._iter}  {rmax}  {self.residual_norm}'
        sprint(fstr, comm=self.comm)
        # if self.options['opt_method'] == 'full' :
        if self.mixer is None :
            rho = self.charge
        else :
            rho = self.mixer(self.prev_charge, self.charge, **kwargs)
            self.charge[:] = rho
            self._format_density_invert(self.charge, self.grid)
        if self.comm.rank > 0 :
            self.residual_norm = 0.0
            self.dp_norm = 0.0
        return self.density

    def get_fermi_level(self, **kwargs):
        results = self.fermi_level
        return results

    def get_forces(self, **kwargs):
        forces = np.zeros((self.subcell.ions.nat, 3))
        return forces

    def get_stress(self, **kwargs):
        pass

    def stop_run(self, **kwargs):
        if self.comm.rank == 0 :
            self.fileobj.close()

    @staticmethod
    def dftpy_opt(ions, rho, pplist, xc_kwargs = None, ke_kwargs = None, stdout = None, options = {}):
        from edftpy.functional import LocalPP, KEDF, Hartree, XC
        from edftpy.evaluator import Evaluator
        from edftpy.density.init_density import AtomicDensity
        #-----------------------------------------------------------------------
        save = environ['STDOUT']
        if stdout is not None :
            environ['STDOUT'] = stdout
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
        evaluator_of = Evaluator(**funcdicts)
        evaluator = partial(evaluator_of.compute, gather = True)
        atomicd = AtomicDensity()
        rho_ini = atomicd.guess_rho(ions, rho.grid)
        #-----------------------------------------------------------------------
        optimization_options = {'econv' : 1e-6 * ions.nat, 'maxfun' : 50, 'maxiter' : 100}
        optimization_options["econv"] *= ions.nat
        optimization_options.update(options)
        opt = Optimization(EnergyEvaluator= evaluator, optimization_options = optimization_options, optimization_method = 'CG-HS')
        new_rho = opt.optimize_rho(guess_rho=rho_ini)
        environ['STDOUT'] = save
        return new_rho
