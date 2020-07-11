import numpy as np
from scipy import signal
import copy
import os
from dftpy.formats.ase_io import ions2ase
import caspytep
import ase.io.castep as ase_io_driver
from ase.calculators.castep import Castep as ase_calc_driver

from ..mixer import LinearMixer, PulayMixer
from ..utils.common import AbsDFT, Grid
from ..utils.math import grid_map_data
from ..density import normalization_density

class CastepKS(AbsDFT):
    """description"""
    def __init__(self, evaluator = None, subcell = None, prefix = 'castep_in_sub', params = None, cell_params = None, 
            exttype = 3, base_in_file = None, castep_in_file = None, mixer = None, ncharge = None, **kwargs):
        '''
        exttype :
                    1 : only pseudo                  : 001 
                    2 : only hartree                 : 010
                    3 : hartree + pseudo             : 011
                    4 : only xc                      : 100
                    5 : pseudo + xc                  : 101
                    6 : hartree + xc                 : 110
                    7 : pseudo + hartree + xc        : 111
        '''
        self.evaluator = evaluator
        self.prefix = prefix
        self.exttype = exttype
        self.subcell = subcell
        self.rho = None
        self.wfs = None
        self.occupations = None
        self.eigs = None
        self.fermi = None
        self.perform_mix = False
        self.mdl = None
        self.ncharge = ncharge
        if base_in_file is None :
            base_in_file = castep_in_file
        if self.prefix :
            self._build_ase_atoms(params, cell_params, base_in_file)
            self._write_cell(self.prefix + '.cell', params = cell_params)
            self._write_params(self.prefix + '.param', params = params)
        else :
            self.prefix = os.path.splitext(base_in_file)[0]

        self._driver_initialise()
        self._iter = 0
        self._filter = None
        self.mixer = mixer
        if 'devel_code' in params :
            self.grid_driver = None
        else :
            self.grid_driver = self.get_grid_driver(self.subcell.grid)
        #-----------------------------------------------------------------------
        self.init_density()
        self.subcell.density[:] = self._format_density_invert(self.charge, self.subcell.grid)
        self.density = self.subcell.density
        self.prev_charge = copy.deepcopy(self.charge)
        #-----------------------------------------------------------------------
        if self.mixer is None :
            # self.mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.2], maxm = 7, coef = [0.2], predecut = 0, delay = 1)
            rho0 = np.mean(self.subcell.density)
            kf = (3.0 * rho0 * np.pi ** 2) ** (1.0 / 3.0)
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, kf, 1.0], maxm = 7, coef = [0.7], predecut = 0, delay = 1)
        if self.grid_driver is not None :
            print('{} has two grids :{} and {}'.format(self.__class__.__name__, self.grid.nr, self.grid_driver.nr))

    @property
    def grid(self):
        return self.subcell.grid

    def get_grid_driver(self, grid):
        current_basis = caspytep.basis.get_current_basis()
        nx = current_basis.ngx
        ny = current_basis.ngy
        nz = current_basis.ngz
        nr = np.array([nx, ny, nz])
        if not np.all(grid.nr == nr):
            grid_driver = Grid(grid.lattice, nr, direct = True)
        else :
            # grid_driver = grid
            grid_driver = None
        return grid_driver

    def _build_ase_atoms(self, params = None, cell_params = None, base_in_file = None):
        ase_atoms = ions2ase(self.subcell.ions)
        ase_atoms.set_calculator(ase_calc_driver())
        self.ase_atoms = ase_atoms
        ase_cell = self.ase_atoms.calc.cell
        if cell_params is not None :
            for k1, v1 in cell_params.items() :
                if isinstance(v1, dict):
                    value = []
                    for k2, v2 in v1.items() :
                        value.append((k2, v2))
                        # ase_cell.__setattr__(k1, (k2, v2))
                    ase_cell.__setattr__(k1, value)
                else :
                    ase_cell.__setattr__(k1, v1)

        if base_in_file is not None :
            fileobj = open(base_in_file, 'r')
            calc = ase_io_driver.read_param(fd=fileobj)
            fileobj.close()
            self.ase_atoms.calc.param = calc.param

        driver_params = self.ase_atoms.calc.param

        if params is not None :
            for k1, v1 in params.items() :
                if isinstance(v1, dict):
                    value = []
                    for k2, v2 in v1.items() :
                        value.append((k2, v2))
                    driver_params.__setattr__(k1, value)
                else :
                    driver_params.__setattr__(k1, v1)

    def _driver_initialise(self, **kwargs):
        caspytep.cell.cell_read(self.prefix)

        current_cell = caspytep.cell.get_current_cell()
        caspytep.ion.ion_read()

        real_charge = float(np.dot(current_cell.num_ions_in_species, current_cell.ionic_charge))
        if self.ncharge is not None :
            real_charge = self.ncharge
        else :
            self.ncharge = real_charge
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

    def init_density(self, rho_ini = None):
        #-----------------------------------------------------------------------
        caspytep.density.density_symmetrise(self.mdl.den)
        #-----------------------------------------------------------------------
        if rho_ini is not None :
            self._format_density(rho_ini, sym = False)

        self.charge = self.mdl.den.real_charge

        density = self._format_density_invert(self.charge, self.grid)
        density = normalization_density(density, ncharge = self.ncharge, grid = self.grid)
        self._format_density(density, sym = False)
        # self.charge = self._format_density_invert(self.mdl.den, self.grid)
        # self.charge = normalization_density(self.charge, ncharge = self.ncharge, grid = self.grid)
        # self._format_density(self.charge, sym = False)
        #-----------------------------------------------------------------------
        # self.mdl.den.real_charge[self.mdl.den.real_charge < 1E-30] = 1E-30
        # print('sum', np.sum(self.mdl.den.real_charge)/np.size(self.mdl.den.real_charge))
        # self.mdl.den.real_charge *= real_charge/ (np.sum(self.mdl.den.real_charge)/np.size(self.mdl.den.real_charge))
        # print('sum2', np.sum(self.mdl.den.real_charge)/np.size(self.mdl.den.real_charge))
        #-----------------------------------------------------------------------

    def _write_cell(self, outfile, pbc = None, params = None, **kwargs):
        ase_atoms = self.ase_atoms
        ase_io_driver.write_cell(outfile, ase_atoms, force_write = True)
        return

    def _write_params(self, outfile, params = None, **kwargs):
        driver_params = self.ase_atoms.calc.param
        ase_io_driver.write_param(outfile, driver_params, force_write = True)
        return

    def _format_density(self, density, volume = None, sym = True, **kwargs):
        #-----------------------------------------------------------------------
        if volume is None :
            volume = density.grid.volume
        if self.grid_driver is not None :
            charge = grid_map_data(density, grid = self.grid_driver) * volume
        else :
            charge = density * volume
        #-----------------------------------------------------------------------
        self.mdl.den.real_charge[:] = charge.ravel(order = 'F')
        if sym :
            caspytep.density.density_symmetrise(self.mdl.den)
            caspytep.density.density_augment(self.mdl.wvfn,self.mdl.occ,self.mdl.den)
        self.charge = self.mdl.den.real_charge
        return

    def _format_density_invert(self, charge = None, grid = None, **kwargs):
        from edftpy.utils.common import Field

        if hasattr(charge, 'real_charge'):
            density = charge.real_charge.copy()
        else :
            density = charge.copy()

        # if charge is None :
            # charge = self.charge
        # density = charge.copy()

        if grid is None :
            grid = self.grid

        if self.grid_driver is not None and np.any(self.grid_driver.nr != grid.nr):
            density /= grid.volume
            density = Field(grid=self.grid_driver, direct=True, data = density, order = 'F')
            rho = grid_map_data(density, grid = grid)
        else :
            density /= grid.volume
            rho = Field(grid=grid, rank=1, direct=True, data = density, order = 'F')
        return rho

    def _get_extpot(self, charge = None, grid = None, **kwargs):
        rho = self._format_density_invert(charge, grid, **kwargs)
        # func = self.evaluator(rho, embed = False)
        # # func.potential *= self.filter
        # extpot = func.potential.ravel(order = 'F')
        # extene = func.energy
        #-----------------------------------------------------------------------
        self.evaluator.get_embed_potential(rho, gaussian_density = self.subcell.gaussian_density, with_global = True)
        extpot = self.evaluator.embed_potential
        extene = (extpot * rho).integral()
        if self.grid_driver is not None :
            extpot = grid_map_data(extpot, grid = self.grid_driver)
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

        extpot, extene = self._get_extpot(self.charge, density.grid)
        self.prev_charge = self.charge.copy()

        if self._iter > 0 :
            # print('perform_mix', self.exttype,self.perform_mix)
            self.mdl.converged = caspytep.electronic.electronic_minimisation_edft_ext(
                self.mdl, extpot, extene, self.exttype, self.perform_mix)
        else :
            self.mdl.converged = caspytep.electronic.electronic_minimisation(self.mdl)
        # self.prev_charge = self.mdl.den.real_charge.copy()

        caspytep.density.density_calculate_soft(self.mdl.wvfn,self.mdl.occ, self.mdl.den)
        if sym :
            caspytep.density.density_symmetrise(self.mdl.den)
            caspytep.density.density_augment(self.mdl.wvfn,self.mdl.occ,self.mdl.den)

        self.charge = self.mdl.den.real_charge
        rho = self._format_density_invert(self.charge, density.grid)
        # rho = self._format_density_invert(self.mdl.den, density.grid)
        print('KS_Chemical_potential ', self.get_fermi_level())
        return rho

    def get_kinetic_energy(self, **kwargs):
        energy = caspytep.electronic.electronic_get_energy('kinetic_energy')
        return energy

    def get_energy(self, density = None, **kwargs):
        total_energy = caspytep.electronic.electronic_get_energy('total_energy')
        ion_ion_energy0 = caspytep.electronic.electronic_get_energy('ion_ion_energy0')
        if density is None :
            density = self._format_density_invert(self.mdl.den, self.grid)
        energy = self.evaluator(density, calcType = ['E'], with_global = False, embed = False).energy
        energy += total_energy - ion_ion_energy0
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], **kwargs):
        func = self.evaluator(density, calcType = ['E'], with_global = False, embed = False)
        etype = 1
        if 'E' in calcType :
            if etype == 1 :
                kinetic_energy = caspytep.electronic.electronic_get_energy('kinetic_energy')
                nonlocal_energy = caspytep.electronic.electronic_get_energy('nonlocal_energy')
                ts = caspytep.electronic.electronic_get_energy('-TS')

                locps_energy = caspytep.electronic.electronic_get_energy('locps_energy')
                ion_noncoulomb_energy = caspytep.electronic.electronic_get_energy('ion_noncoulomb_energy')

                hartree_energy = caspytep.electronic.electronic_get_energy('hartree_energy')

                xc_energy = caspytep.electronic.electronic_get_energy('xc_energy')
                func.energy += kinetic_energy + nonlocal_energy + ts
                if self.exttype & 1 == 0 :
                    func.energy += locps_energy + ion_noncoulomb_energy
                if self.exttype & 2 == 0 :
                    func.energy += hartree_energy
                if self.exttype & 4 == 0 :
                    func.energy += xc_energy
            else :
                # This way need give exact energy of external potential (extene)
                #-----------------------------------------------------------------------
                total_energy = caspytep.electronic.electronic_get_energy('total_energy')
                ion_ion_energy0 = caspytep.electronic.electronic_get_energy('ion_ion_energy0')
                locps_energy = caspytep.electronic.electronic_get_energy('locps_energy')
                hartree_energy = caspytep.electronic.electronic_get_energy('hartree_energy')
                xc_energy = caspytep.electronic.electronic_get_energy('xc_energy')
                ion_noncoulomb_energy = caspytep.electronic.electronic_get_energy('ion_noncoulomb_energy')
                # func.energy += total_energy - ion_ion_energy0 - locps_energy - ion_noncoulomb_energy - hartree_energy
                func.energy += total_energy - ion_ion_energy0
                if self.exttype & 1 == 1 :
                    # print('ks 1', locps_energy + ion_noncoulomb_energy)
                    func.energy -= (locps_energy + ion_noncoulomb_energy)
                if self.exttype & 2 == 2 :
                    func.energy -= hartree_energy
                    # print('ks 2', hartree_energy)
                if self.exttype & 4 == 4 :
                    func.energy -= xc_energy
                    # print('ks 4', xc_energy)
        print('sub_energy_ks', func.energy)
        return func

    def update_density(self, **kwargs):
        mix_grid = False
        # mix_grid = True
        if self.mixer is None :
            if self._iter > 1 :
                caspytep.dm.dm_mix_density(self.mdl.den, self.mdl.den)
                # mixed, residual_norm = caspytep.dm.dm_mix_density(self.mdl.den, self.mdl.den)
                # print('residual_norm', self._iter, residual_norm)
            rho = self._format_density_invert(self.mdl.den, self.grid)
        else :
            if self.grid_driver is not None and mix_grid:
                grid = self.grid_driver
            else :
                grid = self.grid
            prev_density = self._format_density_invert(self.prev_charge, grid)
            density = self._format_density_invert(self.charge, grid)
            # density = self._format_density_invert(self.mdl.den, grid)
            #-----------------------------------------------------------------------
            r = density - prev_density
            print('res_norm_ks', self._iter, np.max(abs(r)), np.sqrt(np.sum(r * r)/np.size(r)))
            #-----------------------------------------------------------------------
            rho = self.mixer(prev_density, density, **kwargs)
            if self.grid_driver is not None and mix_grid:
                rho = grid_map_data(rho, grid = self.grid)
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

    def get_forces(self, icalc = 2, **kwargs):
        """
        icalc :
                0 : all
                1 : no ewald
                2 : no ewald and local_potential
        """
        extpot, extene = self._get_extpot(self.mdl.den, self.grid)
        forces = np.empty((3, self.subcell.ions.nat), order = 'F')
        caspytep.firstd.firstd_calculate_forces_edft_ext(self.mdl, forces, extpot, extene, self.exttype, icalc)
        forces = np.transpose(forces)
        return forces

    def get_stress(self, **kwargs):
        pass
