from ..utils.common import AbsDFT
import numpy as np
from scipy import signal
import copy

import caspytep


class CastepKS(AbsDFT):
    """description"""
    def __init__(self, evaluator = None, prefix = 'castep_in', ions = None, params = None, grid = None, rho_ini = None, exttype = 3, **kwargs):
        '''
        exttype :
                    2 : only hartree
                    3 : hartree and pseudo
        '''
        self.evaluator = evaluator
        self.grid = grid
        self.prefix = prefix
        self.exttype = exttype
        self.rho = None
        self.wfs = None
        self.occupations = None
        self.eigs = None
        self.fermi = None
        self.perform_mix = False
        self.mdl = None
        self.density = None
        if ions is not None :
            self._write_cell(self.prefix + '.cell', ions, params = params)
        self._castep_initialise(rho_ini = rho_ini, **kwargs)
        # self.prev_density = None
        self.prev_density = copy.deepcopy(self.mdl.den)
        self.iter = 0
        self._filter = None

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
        #-----------------------------------------------------------------------
        if rho_ini is not None :
            charge = rho_ini * (rho_ini.grid.dV*np.size(rho_ini))
            mdl.den.real_charge = charge.ravel()
        #-----------------------------------------------------------------------
        current_params.max_scf_cycles = 1
        self.mdl = mdl

    def _write_cell(self, outfile, ions, pbc = None, params = None, **kwargs):
        from dftpy.formats import ase_io
        ase_io.ase_write(outfile, ions, format='castep-cell', pbc=None, positions_frac = True, **kwargs)
        if params is not None :
            with open(outfile, 'a') as fw:
                for line in params :
                    fw.write(line + '\n')
        return

    def _format_density(self, density, dv = None, sym = True, **kwargs):
        if dv is None :
            dv = density.grid.dV
        charge = density * (np.size(density) * dv)
        self.mdl.den.real_charge = charge.ravel(order = 'F')
        if sym :
            caspytep.density.density_symmetrise(self.mdl.den)
            caspytep.density.density_augment(self.mdl.wvfn,self.mdl.occ,self.mdl.den)
        return

    def _format_density_invert(self, charge, grid, **kwargs):
        from edftpy.utils.common import Field

        density = charge.real_charge / (grid.dV*np.size(charge.real_charge))
        rho = Field(grid=grid, rank=1, direct=True, data = density, order = 'F')
        return rho

    def _get_extpot(self, charge, grid, **kwargs):
        rho = self._format_density_invert(charge, grid, **kwargs)
        func = self.evaluator(rho)
        #-----------------------------------------------------------------------
        # potential = func.potential
        # outfile = 'den.dat'
        # with open(outfile, "w") as fw:
            # fw.write("{0[0]:10d} {0[1]:10d} {0[2]:10d}\n".format(potential.grid.nr))
            # size = np.size(potential)
            # nl = size // 3
            # outrho = potential.ravel(order="F")
            # for line in outrho[: nl * 3].reshape(-1, 3):
                # fw.write("{0[0]:22.15E} {0[1]:22.15E} {0[2]:22.15E}\n".format(line))
            # for line in outrho[nl * 3 :]:
                # fw.write("{0:22.15E}".format(line))
        # stop
        pot = func.potential
        np.savetxt('kkk', np.c_[rho.ravel(), pot.ravel()])
        # input('pause')
        # stop
        #-----------------------------------------------------------------------
        # func.potential *= self.filter
        #-----------------------------------------------------------------------
        extpot = func.potential.ravel(order = 'F')
        extene = func.energy
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

        # self.prev_density = copy.deepcopy(self.mdl.den)
        #-----------------------------------------------------------------------
        if self.iter < 2 :
            sym = False
        else :
            sym = True
        # if self.iter > 0 :
            # self.perform_mix = True
        self.iter += 1
        self._format_density(density, sym = sym)
        self.prev_density = copy.deepcopy(self.mdl.den)
        #-----------------------------------------------------------------------
        extpot, extene = self._get_extpot(self.mdl.den, density.grid)
        self.mdl.converged = caspytep.electronic.electronic_minimisation_edft_ext(
                self.mdl, extpot, extene, self.exttype, self.perform_mix)
        caspytep.density.density_calculate_soft(self.mdl.wvfn,self.mdl.occ, self.mdl.den)
        # if sym :
            # caspytep.density.density_symmetrise(self.mdl.den)
            # caspytep.density.density_augment(self.mdl.wvfn,self.mdl.occ,self.mdl.den)
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
        if 'E' in calcType :
            total_energy = caspytep.electronic.electronic_get_energy('total_energy')
            # xc_energy = caspytep.electronic.electronic_get_energy('xc_energy')
            # kinetic_energy = caspytep.electronic.electronic_get_energy('kinetic_energy')
            # nonlocal_energy = caspytep.electronic.electronic_get_energy('nonlocal_energy')
            ion_ion_energy0 = caspytep.electronic.electronic_get_energy('ion_ion_energy0')
            locps_energy = caspytep.electronic.electronic_get_energy('locps_energy')
            hartree_energy = caspytep.electronic.electronic_get_energy('hartree_energy')
            ion_noncoulomb_energy = caspytep.electronic.electronic_get_energy('ion_noncoulomb_energy')
            # func.energy += total_energy - ion_ion_energy0 - locps_energy - 2.0 * ion_noncoulomb_energy - hartree_energy
            func.energy += total_energy - ion_ion_energy0 - locps_energy - ion_noncoulomb_energy - hartree_energy
            # func.energy += kinetic_energy + xc_energy
        return func

    def update_density(self, **kwargs):
        caspytep.dm.dm_mix_density(self.prev_density, self.mdl.den)
        rho = self._format_density_invert(self.mdl.den, self.grid)
        return rho

    def get_fermi_level(self, **kwargs):
        results = self.mdl.fermi_energy
        return results
