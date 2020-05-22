import numpy as np
from scipy import signal
import copy
from dftpy.formats.ase_io import ions2ase
import caspytep
import ase.io.castep as ase_io_castep
import ase.calculators.castep as ase_calc_castep

from ..mixer import LinearMixer, PulayMixer
from ..utils.common import AbsDFT

class CastepKS(AbsDFT):
    """description"""
    def __init__(self, evaluator = None, prefix = 'castep_in_sub', ions = None, params = None, cell_params = None, 
            grid = None, rho_ini = None, exttype = 3, castep_in_file = None, mixer = None, **kwargs):
        '''
        exttype :
                    1 : only pseudo
                    2 : only hartree
                    3 : hartree and pseudo
        '''
        self.evaluator = evaluator
        self.grid = grid
        self.prefix = prefix
        self.exttype = exttype
        self._ions = ions
        self.rho = None
        self.wfs = None
        self.occupations = None
        self.eigs = None
        self.fermi = None
        self.perform_mix = False
        self.mdl = None
        self.density = None
        self._build_ase_atoms(ions, params, cell_params, castep_in_file)
        self._write_cell(self.prefix + '.cell', ions, params = cell_params)
        self._write_params(self.prefix + '.param', params = params)

        self._castep_initialise(rho_ini = rho_ini, **kwargs)
        self.prev_density = copy.deepcopy(self.mdl.den)
        self._iter = 0
        self._filter = None
        self.mixer = mixer
        if self.mixer is None :
            self.mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.2], maxm = 7, coef = [0.2], predecut = 0, delay = 1)
            # self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0], maxm = 7, coef = [0.2], predecut = 0, delay = 1)
            # self.mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.2], maxm = 7, coef = [0.7], predecut = 0, delay = 1)
            # self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.2, 1.0], maxm = 7, coef = [0.3], predecut = 0, delay = 1)
            # self.mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.2], maxm = 7, coef = [1.0], predecut = None, delay = 1)
            # self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.2, 1.0], maxm = 7, coef = [1.0], predecut = 0, delay = 1)
            # self.mixer = LinearMixer(predtype =None, delay = 1, coef = [0.7])
            # self.mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.2], maxm = 5, coef = [1.0])
            # self.mixer = PulayMixer(predtype =None, predcoef = [0.2], maxm = 5, coef = [0.2])
            # self.mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0], maxm = 7, coef = [1.0], predecut = 20.0, delay = 1)

    def _build_ase_atoms(self, ions, params = None, cell_params = None, castep_in_file = None):
        ase_atoms = ions2ase(ions)
        ase_atoms.set_calculator(ase_calc_castep.Castep())
        self.ase_atoms = ase_atoms
        ase_cell = self.ase_atoms.calc.cell
        if cell_params is not None :
            for k1, v1 in cell_params.items() :
                if isinstance(v1, dict):
                    value = []
                    for k2, v2 in v1.items() :
                        value.append((k2, v2))
                    ase_cell.__setattr__(k1, value)
            else :
                ase_cell.__setattr__(k1, v1)

        if castep_in_file is not None :
            calc = ase_io_castep.read_param(castep_in_file)
            self.ase_atoms.calc.param = calc.param

        castep_params = self.ase_atoms.calc.param

        if params is not None :
            for k1, v1 in params.items() :
                if isinstance(v1, dict):
                    value = []
                    for k2, v2 in v1.items() :
                        value.append((k2, v2))
                    castep_params.__setattr__(k1, value)
            else :
                castep_params.__setattr__(k1, v1)

    def _castep_initialise(self, rho_ini = None, **kwargs):
        caspytep.cell.cell_read(self.prefix)

        current_cell = caspytep.cell.get_current_cell()
        caspytep.ion.ion_read()

        real_charge = float(np.dot(current_cell.num_ions_in_species, current_cell.ionic_charge))
        real_num_atoms = current_cell.mixture_weight.sum()
        fixed_cell = current_cell.cell_constraints.max() == 0

        caspytep.parameters.parameters_read(self.prefix, real_charge, real_num_atoms, fixed_cell)
        current_params = caspytep.parameters.get_current_params()

        caspytep.comms.comms_parallel_strategy(current_params.data_distribution,
                                             current_params.nbands,
                                             current_cell.nkpts,
                                             current_params.num_farms_requested,
                                             current_params.num_proc_in_smp)

        caspytep.cell.cell_distribute_kpoints()
        caspytep.ion.ion_initialise()

        caspytep.parameters.parameters_output(caspytep.stdout)
        caspytep.cell.cell_output(caspytep.stdout)

        caspytep.basis.basis_initialise(current_params.cut_off_energy)
        current_params.fine_gmax = (current_params.fine_grid_scale * np.sqrt(2.0*current_params.cut_off_energy))
        caspytep.ion.ion_real_initialise()

        mdl = caspytep.model.model_state()
        current_params.max_scf_cycles = 1
        self.mdl = mdl
        #-----------------------------------------------------------------------
        if rho_ini is not None :
            self._format_density(rho_ini, sym = True)
        #-----------------------------------------------------------------------

    def _write_cell(self, outfile, ions, pbc = None, params = None, **kwargs):
        ase_atoms = self.ase_atoms
        ase_io_castep.write_cell(outfile, ase_atoms, force_write = True)
        return

    def _write_params(self, outfile, params = None, **kwargs):
        castep_params = self.ase_atoms.calc.param
        ase_io_castep.write_param(outfile, castep_params, force_write = True)
        return

    def _format_density(self, density, dv = None, sym = True, **kwargs):
        if dv is None :
            dv = density.grid.dV
        charge = density * (np.size(density) * dv)
        self.mdl.den.real_charge[:] = charge.ravel(order = 'F')
        if sym :
            caspytep.density.density_symmetrise(self.mdl.den)
            caspytep.density.density_augment(self.mdl.wvfn,self.mdl.occ,self.mdl.den)
        return

    def _format_density_invert(self, charge, grid, **kwargs):
        from edftpy.utils.common import Field

        if hasattr(charge, 'real_charge'):
            density = charge.real_charge / (grid.dV*np.size(charge.real_charge))
        else :
            density = charge / (grid.dV*np.size(charge))
        rho = Field(grid=grid, rank=1, direct=True, data = density, order = 'F')
        return rho

    def _get_extpot(self, charge, grid, **kwargs):
        rho = self._format_density_invert(charge, grid, **kwargs)
        # func = self.evaluator(rho)
        # # func.potential *= self.filter
        # extpot = func.potential.ravel(order = 'F')
        # extene = func.energy
        #-----------------------------------------------------------------------
        # self.evaluator.get_embed_potential(rho, with_global = True)
        extpot = self.evaluator.embed_potential
        extene = (extpot * rho).integral()
        extpot = extpot.ravel(order = 'F')
        #-----------------------------------------------------------------------
        return extpot, extene

    def _windows_function(self, grid, alpha = 0.5, bd = [5, 5, 5], **kwargs):
        wf = []
        for i in range(3):
            if grid.pbc[i] :
                wind = np.ones(grid.nr[i])
            else :
                wind = np.zeros(grid.nr[i])
                n = grid.nr[i] - 2 * bd[i]
                wind[bd[i]:n+bd[i]] = signal.tukey(n, alpha)
            wf.append(wind)
        array = np.einsum("i, j, k -> ijk", wf[0], wf[1], wf[2])
        return array

    @property
    def filter(self):
        if self._filter is None :
            self._filter = self._windows_function(self.grid)
        return self._filter

    def get_density(self, density, vext = None, **kwargs):
        '''
        Must call first time
        '''
        if self.grid is None :
            self.grid = density.grid
        #-----------------------------------------------------------------------
        self._iter += 1
        sym = True
        self.perform_mix = True
        if self._iter == 1 :
            caspytep.parameters.get_current_params().max_scf_cycles = 2
            self.perform_mix = False
        else :
            caspytep.parameters.get_current_params().max_scf_cycles = 1
            self.perform_mix = False

        self._format_density(density, sym = sym)

        self.prev_density = copy.deepcopy(self.mdl.den.real_charge[:])
        #-----------------------------------------------------------------------
        extpot, extene = self._get_extpot(self.mdl.den, density.grid)
        if self._iter > 0 :
            # print('perform_mix', self.exttype,self.perform_mix)
            self.mdl.converged = caspytep.electronic.electronic_minimisation_edft_ext(
                self.mdl, extpot, extene, self.exttype, self.perform_mix)
        else :
            self.mdl.converged = caspytep.electronic.electronic_minimisation(self.mdl)

        caspytep.density.density_calculate_soft(self.mdl.wvfn,self.mdl.occ, self.mdl.den)
        if sym :
            caspytep.density.density_symmetrise(self.mdl.den)
            caspytep.density.density_augment(self.mdl.wvfn,self.mdl.occ,self.mdl.den)
        rho = self._format_density_invert(self.mdl.den, density.grid)
        return rho

    def get_kinetic_energy(self, **kwargs):
        energy = caspytep.electronic.electronic_get_energy('kinetic_energy')
        return energy

    def get_energy(self, density = None, **kwargs):
        total_energy = caspytep.electronic.electronic_get_energy('total_energy')
        ion_ion_energy0 = caspytep.electronic.electronic_get_energy('ion_ion_energy0')
        if density is None :
            density = self._format_density_invert(self.mdl.den, self.grid)
        energy = self.evaluator(density, calcType = ['E'], with_global = False).energy
        energy += total_energy - ion_ion_energy0
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], **kwargs):
        func = self.evaluator(density, calcType = ['E'], with_global = False)
        etype = 2
        if 'E' in calcType :
            if etype == 1 :
                xc_energy = caspytep.electronic.electronic_get_energy('xc_energy')
                kinetic_energy = caspytep.electronic.electronic_get_energy('kinetic_energy')
                ts = caspytep.electronic.electronic_get_energy('-TS')
                func.energy += kinetic_energy + xc_energy + 0.5 * ts
            else :
                #-----------------------------------------------------------------------
                total_energy = caspytep.electronic.electronic_get_energy('total_energy')
                # nonlocal_energy = caspytep.electronic.electronic_get_energy('nonlocal_energy')
                ion_ion_energy0 = caspytep.electronic.electronic_get_energy('ion_ion_energy0')
                locps_energy = caspytep.electronic.electronic_get_energy('locps_energy')
                hartree_energy = caspytep.electronic.electronic_get_energy('hartree_energy')
                ion_noncoulomb_energy = caspytep.electronic.electronic_get_energy('ion_noncoulomb_energy')
                func.energy += total_energy - ion_ion_energy0 - locps_energy - ion_noncoulomb_energy - hartree_energy
        return func

    def update_density(self, **kwargs):
        # if self._iter > 0 :
            # mixed, residual_norm = caspytep.dm.dm_mix_density(self.mdl.den, self.mdl.den)
            # print('residual_norm', residual_norm)
        # rho = self._format_density_invert(self.mdl.den, self.grid)
        #-----------------------------------------------------------------------
        prev_density = self._format_density_invert(self.prev_density, self.grid)
        density = self._format_density_invert(self.mdl.den, self.grid)
        #-----------------------------------------------------------------------
        r = density - prev_density
        print('res_norm_ks', self._iter, np.max(abs(r)), np.sqrt(np.sum(r * r)/np.size(r)))
        #-----------------------------------------------------------------------
        rho = self.mixer(prev_density, density, **kwargs)
        #-----------------------------------------------------------------------
        return rho

    def get_fermi_level(self, **kwargs):
        results = self.mdl.fermi_energy
        return results

    def get_energy_part(self, ename, density = None, **kwargs):
        if ename == 'TOTAL' :
            key = 'total_energy'
        elif ename == 'XC' :
            key = 'xc_energy'
        elif ename == 'KEDF' :
            key = 'kinetic_energy'
        elif ename == 'NONLOCAL' :
            key = 'nonlocal_energy'
        elif ename == 'EWALD' :
            key = 'ion_ion_energy0'
        elif ename == 'LOCAL' :
            key = 'locps_energy'
        elif ename == 'HARTREE' :
            key = 'hartree_energy'
        elif ename == 'LOCALC' :
            key = 'ion_noncoulomb_energy'
        else :
            raise AttributeError("!ERROR : not contains this energy", ename)
        energy = caspytep.electronic.electronic_get_energy(key)
        return energy

    def get_forces(self, **kwargs):
        forces = np.empty((3, self._ions.nat))
        caspytep.firstd.firstd_calculate_forces_edft_ext(self.mdl, forces, 2)
        forces = np.transpose(forces)
        return forces
