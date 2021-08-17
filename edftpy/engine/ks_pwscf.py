import qepy

import numpy as np
import copy
import os
import ase.io.espresso as ase_io_driver
from ase.calculators.espresso import Espresso as ase_calc_driver
from collections import OrderedDict

from edftpy.io import ions2ase
from dftpy.constants import LEN_CONV, ENERGY_CONV

from edftpy.mixer import LinearMixer, PulayMixer, AbstractMixer
from edftpy.utils.common import Grid, Field, Functional, Atoms
from edftpy.utils.math import grid_map_data
from edftpy.utils import clean_variables
from edftpy.density import normalization_density
from edftpy.functional import hartree_energy
from edftpy.mpi import sprint, SerialComm, MP
from edftpy.engine.driver import Driver

unit_len = LEN_CONV["Bohr"]["Angstrom"] / qepy.constants.BOHR_RADIUS_SI / 1E10
unit_vol = unit_len ** 3

class PwscfKS(Driver):
    """
    Note :
        The extpot separated into two parts : v.of_r and vltot will be a better and safe way
    """
    def __init__(self, params = None, cell_params = None, diag_conv = 1E-6, **kwargs):
        '''
        Here, prefix is the name of the input file
        exttype :
                    1 : only pseudo                  : 001
                    2 : only hartree                 : 010
                    3 : hartree + pseudo             : 011
                    4 : only xc                      : 100
                    5 : pseudo + xc                  : 101
                    6 : hartree + xc                 : 110
                    7 : pseudo + hartree + xc        : 111
        '''
        kwargs["technique"] = 'KS'
        super().__init__(**kwargs)

        self._input_ext = '.in'
        if self.prefix :
            if self.comm.rank == 0 :
                self.build_input(params, cell_params, self.base_in_file)
        else :
            self.prefix, self._input_ext= os.path.splitext(self.base_in_file)

        if self.comm.size > 1 : self.comm.Barrier()
        self._grid = None
        self._grid_sub = None
        self.outfile = self.prefix + '.out'
        self._driver_initialise(append = self.append)
        self.grid_driver = self.get_grid_driver(self.grid)
        #-----------------------------------------------------------------------
        self.atmp = np.zeros(1)
        self.atmp2 = np.zeros((1, self.nspin), order='F')
        #-----------------------------------------------------------------------
        self.mix_driver = None
        if self.mixer is None :
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, 0.6, 1.0], maxm = 7, coef = 0.5, predecut = 0, delay = 1)
        elif isinstance(self.mixer, float):
            self.mix_driver = self.mixer

        fstr = f'Subcell grid({self.prefix}): {self.subcell.grid.nrR}  {self.subcell.grid.nr}\n'
        fstr += f'Subcell shift({self.prefix}): {self.subcell.grid.shift}\n'
        if self.grid_driver is not None :
            fstr += f'{self.prefix} has two grids :{self.grid.nrR} and {self.grid_driver.nrR}'
        sprint(fstr, comm=self.comm, level=1)
        qepy.qepy_mod.qepy_write_stdout(fstr)
        #-----------------------------------------------------------------------
        self.init_density()
        self.embed = qepy.qepy_common.embed_base()
        self.embed.diag_conv = diag_conv
        self.update_workspace(first = True, restart = self.restart)

    def update_workspace(self, subcell = None, first = False, update = 0, restart = False, **kwargs):
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
        if isinstance(self.mixer, AbstractMixer):
            self.mixer.restart()
        if subcell is not None :
            self.subcell = subcell

        self.gaussian_density = self.get_gaussian_density(self.subcell, grid = self.grid)

        if not first :
            pos = self.subcell.ions.pos.to_cart().T / self.subcell.grid.latparas[0]
            qepy.qepy_mod.qepy_update_ions(self.embed, pos, update)
            if update == 0 :
                # get new density
                qepy.qepy_mod.qepy_get_rho(self.charge)
                self.density[:] = self._format_density_invert()
        if self.task == 'optical' :
            if first :
                self.embed.tddft.initial = True
                self.embed.tddft.finish = False
                self.embed.tddft.nstep = 900000 # Any large enough number
                qepy.qepy_tddft_readin(self.prefix + self._input_ext)
                qepy.qepy_tddft_main_setup(self.embed)
                if restart :
                    qepy.qepy_tddft_mod.qepy_cetddft_wfc2rho()
                    # get new density
                    qepy.qepy_mod.qepy_get_rho(self.charge)
                    self.density[:] = self._format_density_invert()

        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        if self.comm.rank == 0 :
            core_charge = np.empty(grid.nnr, order = 'F')
        else :
            core_charge = self.atmp
        qepy.qepy_mod.qepy_get_rho_core(core_charge)
        self.core_density= self._format_density_invert(core_charge)

        self.core_density_sub = Field(grid = self.grid_sub)
        self.grid_sub.scatter(self.core_density, out = self.core_density_sub)
        #-----------------------------------------------------------------------
        # if self.ncharge is not None :
        #     nelec = qepy.klist.get_nelec()
        #     # if self.comm.rank == 0 :
        #     #     self.density[:] *= (nelec + self.ncharge)/ nelec
        #     if self.exttype < 0 :
        #         if self.comm.rank == 0 :
        #             self.density[:] = nelec / self.grid.volume
        #         self.core_density[:] = 0.0
        #         self.core_density_sub[:] = 0.0
        #         core_charge[:] = 0.0
        #         qepy.qepy_mod.qepy_set_rho_core(core_charge)
        #         self._format_density()
        #         if self.gaussian_density is not None :
        #             self.gaussian_density_inter = self.gaussian_density.copy()
        #             self.gaussian_density[:] = 0.0
        #         else :
        #             self.gaussian_density_inter = None
        #         self.embed.nlpp = True
        #     # qepy.klist.set_tot_charge(self.ncharge)
        #     # qepy.klist.set_nelec(nelec+self.ncharge)
        #     # self._format_density()
        # if self.comm.rank == 0 :
        #     print('ncharge_sub', self.density.integral())
        #-----------------------------------------------------------------------

        if self.comm.rank == 0 :
            clean_variables(core_charge)
        return

    @property
    def grid(self):
        if self._grid is None :
            if np.all(self.subcell.grid.nrR == self.subcell.grid.nr):
                self._grid = self.subcell.grid
            else :
                self._grid = Grid(self.subcell.grid.lattice, self.subcell.grid.nrR, direct = True)
        return self._grid

    @property
    def grid_sub(self):
        if self._grid_sub is None :
            mp = MP(comm = self.comm, decomposition = self.subcell.grid.mp.decomposition)
            self._grid_sub = Grid(self.subcell.grid.lattice, self.subcell.grid.nrR, direct = True, mp = mp)
        return self._grid_sub

    def get_grid_driver(self, grid):
        nr = np.zeros(3, dtype = 'int32')
        qepy.qepy_mod.qepy_get_grid(nr)
        if not np.all(grid.nrR == nr) and self.comm.rank == 0 :
            grid_driver = Grid(grid.lattice, nr, direct = True)
        else :
            grid_driver = None
        return grid_driver

    def build_input(self, params, cell_params, base_in_file):
        in_params, cards = self._build_ase_atoms(params, cell_params, base_in_file, ions = self.subcell.ions, prefix = self.prefix)
        self._write_params(self.prefix + self._input_ext, params = in_params, cell_params = cell_params, cards = cards)

    def _fix_params(self, params = None, prefix = 'sub_'):
        #output of qe
        default_params = OrderedDict({
                'control' :
                {
                    'calculation' : 'scf',
                    # 'verbosity' : 'high',
                    'restart_mode' : 'from_scratch',
                    # 'iprint' : 1,
                    'disk_io' : 'none', # do not save anything for qe
                    },
                'system' :
                {
                    'ibrav' : 0,
                    'nat' : 1,
                    'ntyp' : 1,
                    # 'ecutwfc' : 40,
                    # 'nosym' : True,
                    'occupations' : 'smearing',
                    'degauss' : 0.001,
                    'smearing' : 'gaussian',
                    },
                'electrons' :
                {
                    'mixing_beta' : 0.5,
                    },
                'ions' :
                {
                    'pot_extrapolation' : 'atomic',  # extrapolation for potential from preceding ionic steps
                    'wfc_extrapolation' : 'none'   # no extrapolation for wfc from preceding ionic steps
                    },
                'cell' : {}
                })
        fix_params = {
                'control' :
                {
                    'prefix' : prefix,
                    'outdir' : prefix + '.tmp'  # outdir of qe
                    },
                'electrons' :
                {
                    'electron_maxstep' : 1, # only run 1 step for scf
                    'conv_thr' : 0.0, # set very high accuracy, otherwise pwscf mixer cause davcio (15) error
                    },
                }
        if not params :
            params = default_params.copy()

        for k1, v1 in fix_params.items() :
            if k1 not in params :
                params[k1] = {}
            for k2, v2 in v1.items() :
                params[k1][k2] = v2
        return params

    def _build_ase_atoms(self, params = None, cell_params = None, base_in_file = None, ions = None, prefix = 'sub_'):
        ase_atoms = ions2ase(ions)
        ase_atoms.set_calculator(ase_calc_driver())
        self.ase_atoms = ase_atoms

        if base_in_file :
            fileobj = open(base_in_file, 'r')
            in_params, card_lines = ase_io_driver.read_fortran_namelist(fileobj)
            fileobj.close()
        else :
            in_params = {}
            card_lines = []

        in_params = self._fix_params(in_params, prefix = prefix)

        for k1, v1 in params.items() :
            if k1 not in in_params :
                in_params[k1] = {}
            for k2, v2 in v1.items() :
                in_params[k1][k2] = v2

        return in_params, card_lines

    def _driver_initialise(self, append = False, **kwargs):
        if self.comm is None or isinstance(self.comm, SerialComm):
            comm = None
        else :
            comm = self.comm.py2f()
            # print('comm00', comm, self.comm.size)
        if self.comm.rank == 0 :
            qepy.qepy_mod.qepy_set_stdout(self.outfile, append = append)
        if self.task == 'optical' :
            qepy.qepy_tddft_main_initial(self.prefix + self._input_ext, comm)
            qepy.read_file()
        else :
            qepy.qepy_pwscf(self.prefix + self._input_ext, comm)

    def init_density(self, rho_ini = None):

        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        if self.comm.rank == 0 :
            self.density = Field(grid=self.grid)
            self.prev_density = Field(grid=self.grid)
            self.charge = np.empty((grid.nnr, self.nspin), order = 'F')
            self.prev_charge = np.empty((grid.nnr, self.nspin), order = 'F')
        else :
            self.density = self.atmp
            self.prev_density = self.atmp
            self.charge = self.atmp2
            self.prev_charge = self.atmp2

        self.density_sub = Field(grid = self.grid_sub)
        self.gaussian_density_sub = Field(grid = self.grid_sub)

        if rho_ini is None :
            nc = np.sum(self.subcell.density)
            if nc > -1E-6 :
                rho_ini = self.subcell.density.gather(grid = self.grid)

        if rho_ini is not None :
            if self.comm.rank == 0 : self.density[:] = rho_ini
            self._format_density()
        else :
            qepy.qepy_mod.qepy_get_rho(self.charge)
            self.density[:] = self._format_density_invert()

        # if self.comm.rank == 0 :
            # self.density = normalization_density(self.density, ncharge = self.ncharge, grid = self.grid)
        # self._format_density()

    def _write_params(self, outfile, params = None, cell_params = {}, cards = None, **kwargs):
        cell_params.update(kwargs)
        if 'pseudopotentials' not in cell_params :
            raise AttributeError("!!!ERROR : Must give the pseudopotentials")
        value = {}
        for k2, v2 in cell_params['pseudopotentials'].items():
            value[k2] = os.path.basename(v2)
        cell_params['pseudopotentials'] = value
        # For QE, all pseudopotentials should at same directory
        params['control']['pseudo_dir'] = os.path.dirname(v2)
        if self.ncharge is not None :
            params['system']['tot_charge'] = self.ncharge
        fileobj = open(outfile, 'w')
        if 'kpts' not in cell_params :
            self._update_kpoints(cell_params, cards)
        pw_items = ['control', 'system', 'electrons', 'ions', 'cell']
        pw_params = params.copy()
        for k in params :
            if k not in pw_items : del pw_params[k]
        #-----------------------------------------------------------------------
        rm_items = ['ibrav', 'celldm(1)', 'celldm(2)', 'celldm(3)', 'celldm(4)', 'celldm(5)', 'celldm(6)']
        for item in rm_items :
            if item in pw_params['system'] :
                del pw_params['system'][item]
        #-----------------------------------------------------------------------
        ase_io_driver.write_espresso_in(fileobj, self.ase_atoms, pw_params, **cell_params)
        self._write_params_cards(fileobj, params, cards)
        self._write_params_others(fileobj, params)
        fileobj.close()
        return

    def _update_kpoints(self, cell_params, cards = None):
        if cards is None or len(cards) == 0 :
            return
        lines = iter(cards)
        for line in lines :
            if line.split()[0].upper() == 'K_POINTS' :
                ktype = line.split()[1].lower()
                if 'gamma' in ktype :
                    break
                elif 'automatic' in ktype :
                    line = next(lines)
                    item = list(map(int, line.split()))
                    d = {'kpts' : item[:3], 'koffset' : item[3:6]}
                    cell_params.update(d)
                else :
                    raise AttributeError("Not supported by ASE")
                break
        return

    def _write_params_cards(self, fd, params = None, cards = None, **kwargs):
        if cards is None or len(cards) == 0 :
            return
        lines = iter(cards)
        items = ['CONSTRAINTS', 'OCCUPATIONS', 'ATOMIC_FORCES']
        for line in lines :
            if line.split()[0] in items :
                fd.write('\n' + line + '\n')
                for line in lines :
                    if not line[0] == '#' and line.split()[0].isupper():
                        break
                    else :
                        fd.write(line + '\n')
        return

    def _write_params_others(self, fd, params = None, **kwargs):
        if params is None or len(params) == 0 :
            return
        fstrl = []
        prefix = self.prefix
        # pw_keys = ['control', 'system', 'electrons', 'ions', 'cell']
        keys = ['inputtddft']
        for section in params:
            if section not in keys : continue
            fstrl.append('&{0}\n'.format(section.upper()))
            for key, value in params[section].items():
                #-----------------------------------------------------------------------
                if key == 'prefix' :
                    value = prefix
                elif key == 'tmp_dir' :
                    value = prefix + '.tmp/'
                #-----------------------------------------------------------------------
                if value is True:
                    fstrl.append('   {0:40} = .true.\n'.format(key))
                elif value is False:
                    fstrl.append('   {0:40} = .false.\n'.format(key))
                else:
                    fstrl.append('   {0:40} = {1!r:}\n'.format(key, value))
            fstrl.append('/\n\n')
        fd.write(''.join(fstrl))
        return

    def _format_density(self, volume = None, sym = False, **kwargs):
        self.prev_charge, self.charge = self.charge, self.prev_charge
        if self.grid_driver is not None and self.comm.rank == 0:
            charge = grid_map_data(self.density, grid = self.grid_driver)
        else :
            charge = self.density
        #-----------------------------------------------------------------------
        charge = charge.reshape((-1, self.nspin), order='F') / unit_vol
        self.charge[:] = charge
        qepy.qepy_mod.qepy_set_rho(charge)
        # if sym :
            # qepy.qepy_sum_band_sym()
            # qepy.qepy_mod.qepy_get_rho(self.charge)
            # self.density[:] = self._format_density_invert(**kwargs)
        #-----------------------------------------------------------------------
        self.prev_density[:] = self.density
        return

    def _format_density_invert(self, charge = None, grid = None, **kwargs):
        if self.comm.rank > 0 : return self.atmp
        if charge is None :
            charge = self.charge

        if grid is None :
            grid = self.grid

        if self.grid_driver is not None and np.any(self.grid_driver.nrR != grid.nrR):
            density = Field(grid=self.grid_driver, direct=True, data = charge, order = 'F')
            rho = grid_map_data(density, grid = grid)
        else :
            rho = Field(grid=grid, rank=1, direct=True, data = charge, order = 'F')
        rho *= unit_vol
        return rho

    def _get_extpot_serial(self, with_global = True, **kwargs):
        if self.comm.rank == 0 :
            self.evaluator.get_embed_potential(self.density, gaussian_density = self.gaussian_density, with_global = with_global)
            extpot = self.evaluator.embed_potential
        else :
            extpot = self.atmp
        return extpot

    def _get_extpot_mpi(self, with_global = True, **kwargs):
        self.grid_sub.scatter(self.density, out = self.density_sub)
        if self.gaussian_density is not None :
            self.grid_sub.scatter(self.gaussian_density, out = self.gaussian_density_sub)
        self.evaluator.get_embed_potential(self.density_sub, gaussian_density = self.gaussian_density_sub, gather = True, with_global = with_global)
        extpot = self.evaluator.embed_potential
        return extpot

    def get_extpot(self, extpot = None, mapping = True, **kwargs):
        if extpot is None :
            # extpot = self._get_extpot_serial(**kwargs)
            extpot = self._get_extpot_mpi(**kwargs)
        if mapping :
            if self.grid_driver is not None and self.comm.rank == 0:
                extpot = grid_map_data(extpot, grid = self.grid_driver)
            extpot = extpot.ravel(order = 'F') * 2.0 # a.u. to Ry
        return extpot

    def _get_extene(self, extpot, **kwargs):
        if self.comm.rank == 0 :
            extene = (extpot * self.density).integral()
        else :
            extene = 0.0

        if self.comm.size > 1 :
            extene = self.comm.bcast(extene, root = 0)
        return extene

    def _get_charge(self):
        #-----------------------------------------------------------------------
        printout = 2
        exxen = 0.0
        #-----------------------------------------------------------------------
        if self.task == 'optical' :
            qepy.qepy_molecule_optical_absorption(self.embed)
        else :
            qepy.qepy_electrons_scf(printout, exxen, self.embed)

        qepy.qepy_mod.qepy_get_rho(self.charge)
        return self.charge

    def get_density(self, vext = None, sdft = 'sdft', **kwargs):
        '''
        '''
        self._iter += 1
        if self._iter == 1 :
            # The first step density mixing also need keep initial = True
            initial = True
        else :
            initial = False

        if self.mix_driver is not None :
            # If use pwscf mixer, do not need format the density
            self.prev_density[:] = self.density
        else :
            self._format_density()

        if sdft == 'pdft' :
            extpot = self.evaluator.embed_potential
            extpot = self.get_extpot(extpot, mapping = True)
            self.embed.lewald = False
        else :
            extpot = self.get_extpot()

        self.prev_charge[:] = self.charge

        qepy.qepy_mod.qepy_set_extpot(self.embed, extpot)
        self.embed.exttype = self.exttype
        self.embed.initial = initial
        self.embed.mix_coef = -1.0
        self.embed.finish = False
        #
        self._get_charge()
        #
        self.energy = self.embed.etotal
        self.dp_norm = self.embed.dnorm
        self.density[:] = self._format_density_invert()
        return self.density

    def get_energy(self, olevel = 0, sdft = 'sdft', **kwargs):
        if olevel == 0 :
            # self._format_density() # Here, we directly use saved density
            if sdft == 'pdft' :
                extpot = self.evaluator.embed_potential
                extpot = self.get_extpot(extpot, mapping = True)
            else :
                extpot = self.get_extpot()
            qepy.qepy_mod.qepy_set_extpot(self.embed, extpot)
            qepy.qepy_calc_energies(self.embed)
        energy = self.embed.etotal * 0.5
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], olevel = 1, sdft = 'sdft', **kwargs):
        if 'E' in calcType :
            energy = self.get_energy(olevel = olevel, sdft = sdft)

            if olevel == 0 :
                self.grid_sub.scatter(density, out = self.density_sub)
                edict = self.evaluator(self.density_sub, calcType = ['E'], with_global = False, with_embed = False, gather = True, split = True)
                func = edict.pop('TOTAL')
            else : # elif olevel == 1 :
                if self.comm.rank == 0 :
                    func = self.evaluator(density, calcType = ['E'], with_global = False, with_embed = True)
                else :
                    func = Functional(name = 'ZERO', energy=0.0, potential=None)

            func.energy += energy
            if sdft == 'sdft' and self.exttype == 0 : func.energy = 0.0
            if self.comm.rank > 0 : func.energy = 0.0

            if olevel == 0 :
                fstr = format("Energy information", "-^80") + '\n'
                for key, item in edict.items():
                    fstr += "{:>12s} energy: {:22.15E} (eV) = {:22.15E} (a.u.)\n".format(key, item.energy* ENERGY_CONV["Hartree"]["eV"], item.energy)
                fstr += "-" * 80 + '\n'
            else :
                fstr = ''
            fstr += f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm, level=1)
            qepy.qepy_mod.qepy_write_stdout(fstr)
            self.energy = func.energy
        return func

    def update_density(self, coef = None, mix_grid = False, **kwargs):
        if self.comm.rank == 0 :
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
            fstr = f'res_norm({self.prefix}): {self._iter}  {rmax}  {self.residual_norm}'
            sprint(fstr, comm=self.comm, level=1)
            qepy.qepy_mod.qepy_write_stdout(fstr)
            #-----------------------------------------------------------------------
            if self.mix_driver is None :
                self.dp_norm = hartree_energy(r)
                rho = self.mixer(prev_density, density, **kwargs)
                if self.grid_driver is not None and mix_grid:
                    rho = grid_map_data(rho, grid = self.grid)
                self.density[:] = rho

        if self.mix_driver is not None :
            if coef is None : coef = self.mix_driver
            self.embed.mix_coef = coef
            qepy.qepy_electrons_scf(2, 0, self.embed)
            self.dp_norm = self.embed.dnorm
            qepy.qepy_mod.qepy_get_rho(self.charge)
            self.density[:] = self._format_density_invert()

        if self.comm.rank > 0 :
            self.residual_norm = 0.0
            self.dp_norm = 0.0
        # if self.comm.size > 1 : self.residual_norm = self.comm.bcast(self.residual_norm, root=0)
        return self.density

    def get_fermi_level(self, **kwargs):
        results = qepy.ener.get_ef()
        return results

    def get_forces(self, icalc = 2, **kwargs):
        """
        icalc :
                0 : all
                1 : no ewald
                2 : no ewald and local_potential
        """
        qepy.qepy_forces(icalc)
        # forces = qepy.force_mod.force.T
        forces = qepy.force_mod.get_array_force().T / 2.0  # Ry to a.u.
        return forces

    def get_stress(self, **kwargs):
        pass

    def end_scf(self, **kwargs):
        if self.task == 'optical' :
            self.embed.tddft.finish = True
            qepy.qepy_molecule_optical_absorption(self.embed)
        else :
            self.embed.finish = True
            qepy.qepy_electrons_scf(0, 0, self.embed)

    def save(self, save = ['D'], **kwargs):
        if 'W' in save :
            what = 'all'
        else :
            what = 'config-nowf'
        qepy.punch(what)
        qepy.close_files(False)

    def stop_run(self, status = 0, save = ['D'], **kwargs):
        if self.task == 'optical' :
            qepy.qepy_stop_tddft(status)
        else :
            if 'W' in save :
                what = 'all'
            else :
                what = 'config-nowf'
            qepy.qepy_stop_run(status, what = what)
        qepy.qepy_mod.qepy_close_stdout(self.outfile)
        qepy.qepy_clean_saved()

    @staticmethod
    def get_ions_from_pw():
        alat = qepy.cell_base.get_alat()
        lattice = qepy.cell_base.get_array_at() * alat
        pos = qepy.ions_base.get_array_tau().T * alat
        atm = qepy.ions_base.get_array_atm()
        ityp = qepy.ions_base.get_array_ityp()
        nat = qepy.ions_base.get_nat()
        symbols = [1]
        labels = []
        for i in range(atm.shape[-1]):
            s = atm[:,i].view('S3')[0].strip().decode("utf-8")
            symbols.append(s)

        for i in range(nat):
            labels.append(symbols[ityp[i]])

        ions = Atoms(labels=labels, pos=pos, cell=lattice, basis="Cartesian")
        return ions

    @staticmethod
    def get_density_from_pw(ions, comm = None):
        nr = np.zeros(3, dtype = 'int32')
        qepy.qepy_mod.qepy_get_grid(nr)

        if comm is None or comm.rank == 0 :
            rho = np.zeros((np.prod(nr), 1), order = 'F')
        else :
            rho = np.zeros(3)

        qepy.qepy_mod.qepy_get_rho(rho)

        if comm is None or comm.rank == 0 :
            grid = Grid(ions.pos.cell.lattice, nr)
            density = Field(grid=grid, direct=True, data = rho, order = 'F')
        else :
            density = np.zeros(1)

        return density

class QEKS(PwscfKS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
