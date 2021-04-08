import caspytep

import numpy as np
from scipy import signal
import copy
import os
from dftpy.formats.ase_io import ions2ase
import ase.io.castep as ase_io_driver
from ase.calculators.castep import Castep as ase_calc_driver

from ..mixer import LinearMixer, PulayMixer
from ..utils.common import Grid, Field
from ..utils.math import grid_map_data
from ..density import normalization_density

from .driver import Driver
from edftpy.mpi import sprint

class CastepKS(Driver):
    """description"""
    def __init__(self, evaluator = None, subcell = None, prefix = 'castep_in_sub', params = None, cell_params = None,
            exttype = 3, base_in_file = None, mixer = None, ncharge = None, options = None, comm = None, **kwargs):
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
        super().__init__(options = options, technique = 'KS')

        self.evaluator = evaluator
        self.exttype = exttype
        self.subcell = subcell
        self.prefix = prefix
        self.ncharge = ncharge
        self.comm = self.subcell.grid.mp.comm
        self.outfile = None
        if self.prefix :
            if self.comm.rank == 0 :
                self.build_input(params, cell_params, base_in_file)
        else :
            self.prefix = os.path.splitext(base_in_file)[0]

        if self.comm.size > 1 : self.comm.Barrier()
        self._driver_initialise()
        self.mixer = mixer

        self._grid = None
        self.grid_driver = self.get_grid_driver(self.grid)
        #-----------------------------------------------------------------------
        self.init_density()
        #-----------------------------------------------------------------------
        self.mix_driver = None
        if self.mixer is None :
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, 0.6, 1.0], maxm = 7, coef = 0.5, predecut = 0, delay = 1)
        elif isinstance(self.mixer, float):
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, 0.6, 1.0], maxm = 7, coef = self.mixer, predecut = 0, delay = 1)
        if self.grid_driver is not None :
            sprint('{} has two grids :{} and {}'.format(self.__class__.__name__, self.grid.nr, self.grid_driver.nr))
        #-----------------------------------------------------------------------
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
        if hasattr(self.mixer, 'restart'):
            self.mixer.restart()
        if subcell is not None :
            self.subcell = subcell
        self.gaussian_density = self.get_gaussian_density(self.subcell, grid = self.grid)
        if not first :
            raise AttributeError("Will implemented soon")
        return

    @property
    def grid(self):
        if self._grid is None :
            if np.all(self.subcell.grid.nrR == self.subcell.grid.nr):
                self._grid = self.subcell.grid
            else :
                self._grid = Grid(self.subcell.grid.lattice, self.subcell.grid.nrR, direct = True)
        return self._grid

    def get_grid_driver(self, grid):
        current_basis = caspytep.basis.get_current_basis()
        nx = current_basis.ngx
        ny = current_basis.ngy
        nz = current_basis.ngz
        nr = np.array([nx, ny, nz])
        if not np.all(grid.nr == nr):
            grid_driver = Grid(grid.lattice, nr, direct = True)
        else :
            grid_driver = None
        return grid_driver

    def build_input(self, params, cell_params, base_in_file):
        self._build_ase_atoms(params, cell_params, base_in_file)
        self._write_cell(self.prefix + '.cell', params = cell_params)
        self._write_params(self.prefix + '.param', params = params)

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
        self.density = Field(grid=self.grid)
        self.prev_density = Field(grid=self.grid)
        #-----------------------------------------------------------------------
        caspytep.density.density_symmetrise(self.mdl.den)
        #-----------------------------------------------------------------------
        if rho_ini is not None :
            self._format_density(rho_ini, sym = False)

        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        # self.charge = np.empty((grid.nnr, 1), order = 'F')
        # self.prev_charge = np.empty((grid.nnr, 1), order = 'F')
        self.charge = np.empty(grid.nnr, order = 'F')
        self.prev_charge = np.empty(grid.nnr, order = 'F')
        self.charge[:] = self.mdl.den.real_charge

        if self.comm.rank == 0 :
            self.density[:] = self._format_density_invert()
            self.density = normalization_density(self.density, ncharge = self.ncharge, grid = self.grid)
        self._format_density(sym = False)

    def _write_cell(self, outfile, pbc = None, params = None, **kwargs):
        ase_atoms = self.ase_atoms
        ase_io_driver.write_cell(outfile, ase_atoms, force_write = True)
        return

    def _write_params(self, outfile, params = None, **kwargs):
        driver_params = self.ase_atoms.calc.param
        ase_io_driver.write_param(outfile, driver_params, force_write = True)
        return

    def _format_density(self, volume = None, sym = True, **kwargs):
        #-----------------------------------------------------------------------
        if volume is None :
            volume = self.density.grid.volume
        # self.prev_density[:] = self.density
        self.prev_charge, self.charge = self.charge, self.prev_charge
        if self.grid_driver is not None :
            charge = grid_map_data(self.density, grid = self.grid_driver) * volume
        else :
            charge = self.density * volume
        #-----------------------------------------------------------------------
        self.mdl.den.real_charge[:] = charge.ravel(order = 'F')
        if sym :
            caspytep.density.density_symmetrise(self.mdl.den)
            caspytep.density.density_augment(self.mdl.wvfn,self.mdl.occ,self.mdl.den)
        self.charge[:] = self.mdl.den.real_charge
        #-----------------------------------------------------------------------
        self.density[:] = self._format_density_invert(**kwargs)
        self.prev_density[:] = self.density
        return

    def _format_density_invert(self, charge = None, grid = None, **kwargs):
        if charge is None :
            charge = self.charge

        if grid is None :
            grid = self.grid

        density = charge/grid.volume
        if self.grid_driver is not None and np.any(self.grid_driver.nrR != grid.nrR):
            density = Field(grid=self.grid_driver, direct=True, data = charge, order = 'F')
            rho = grid_map_data(density, grid = grid)
        else :
            rho = Field(grid=grid, rank=1, direct=True, data = charge, order = 'F')
        return rho

    def _get_extpot(self, **kwargs):
        if self.comm.rank == 0 :
            self.evaluator.get_embed_potential(self.density, gaussian_density = self.gaussian_density)
            extpot = self.evaluator.embed_potential
            extene = (extpot * self.density).integral()
        else :
            extpot = np.empty(self.grid.nrR)
            extene = 0.0
        if self.comm.size > 1 :
            extene = self.comm.bcast(extene, root = 0)
        if self.grid_driver is not None :
            extpot = grid_map_data(extpot, grid = self.grid_driver)
        extpot = extpot.ravel(order = 'F')
        return extpot, extene

    def get_density(self, vext = None, **kwargs):
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

        self._format_density(sym = sym)
        extpot, extene = self._get_extpot()

        self.prev_charge[:] = self.charge

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

        self.charge[:] = self.mdl.den.real_charge
        self.density[:] = self._format_density_invert()
        return self.density

    def get_kinetic_energy(self, **kwargs):
        energy = caspytep.electronic.electronic_get_energy('kinetic_energy')
        return energy

    def get_energy(self, density = None, **kwargs):
        total_energy = caspytep.electronic.electronic_get_energy('total_energy')
        ion_ion_energy0 = caspytep.electronic.electronic_get_energy('ion_ion_energy0')
        if density is None :
            density = self._format_density_invert(self.mdl.den, self.grid)
        energy = self.evaluator(density, calcType = ['E'], with_global = False, with_embed = False).energy
        energy += total_energy - ion_ion_energy0
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], **kwargs):
        func = self.evaluator(density, calcType = ['E'], with_global = False, with_embed = False)
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
        if self.mixer is None :
            if self._iter > 1 :
                self.residual_norm = 0.0
                caspytep.dm.dm_mix_density(self.mdl.den, self.mdl.den, residual_norm = self.residual_norm)
            self.density[:] = self._format_density_invert(self.mdl.den.real_charge, self.grid)
            sprint('res_norm_ks', self._iter, self.residual_norm, self.residual_norm, comm=self.comm)
            return self.density
        if self.comm.rank == 0 :
            mix_grid = False
            # mix_grid = True
            if self.grid_driver is not None and mix_grid:
                prev_density = self._format_density_invert(self.prev_charge, self.grid_driver)
                density = self._format_density_invert(self.charge, self.grid_driver)
            else :
                prev_density = self.prev_density
                density = self.density
            #-----------------------------------------------------------------------
            r = density - prev_density
            self.residual_norm = np.sqrt(np.sum(r * r)/r.size)
            rmax = r.amax()
            sprint('res_norm_ks', self._iter, rmax, self.residual_norm, comm=self.comm)
            #-----------------------------------------------------------------------
            rho = self.mixer(prev_density, density, **kwargs)
            if self.grid_driver is not None and mix_grid:
                rho = grid_map_data(rho, grid = self.grid)
            self.density[:] = rho
        else :
            self.residual_norm = 100.0
        # if self.comm.size > 1 : self.residual_norm = self.comm.bcast(self.residual_norm, root=0)
        if self.comm.rank > 0 : self.residual_norm = 0.0
        return self.density

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
        #-----------------------------------------------------------------------
        # labels = self.subcell.ions.labels
        labels = self.subcell.ions.Z
        u, index, counts = np.unique(labels, return_index=True, return_counts=True)
        sidx = np.argsort(index)
        nats = counts[sidx]
        #-----------------------------------------------------------------------
        # nats = []
        # keys = list(set(labels))
        # keys.sort(key = labels.index)
        # for key in keys :
            # nats.append(labels.count(key))
        #-----------------------------------------------------------------------
        ntyp = len(nats)
        nmax = max(nats)
        fs = np.zeros((3, nmax, ntyp), order = 'F')
        caspytep.firstd.firstd_calculate_forces_edft_ext(self.mdl, fs, extpot, extene, self.exttype, icalc)
        forces = np.empty((self.subcell.ions.nat, 3), order = 'F')
        n = 0
        for i, item in enumerate(nats):
            forces[n:n+item] = fs[:, :item, sidx[i]].T
            n += item
        return forces

    def get_stress(self, **kwargs):
        pass
