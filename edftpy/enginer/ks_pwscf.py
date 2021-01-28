import pwscfpy

import numpy as np
import copy
import os
import ase.io.espresso as ase_io_driver
from ase.calculators.espresso import Espresso as ase_calc_driver
from collections import OrderedDict

from dftpy.formats.ase_io import ions2ase
from dftpy.constants import LEN_CONV, ENERGY_CONV

from edftpy.mixer import LinearMixer, PulayMixer, AbstractMixer
from edftpy.utils.common import Grid, Field, Functional
from edftpy.utils.math import grid_map_data
from edftpy.utils import clean_variables
from edftpy.density import normalization_density
from edftpy.enginer.driver import Driver
from edftpy.hartree import hartree_energy
from edftpy.mpi import sprint, SerialComm, MP

from pwscfpy import constants as pwc
unit_len = LEN_CONV["Bohr"]["Angstrom"] / pwc.BOHR_RADIUS_SI / 1E10
unit_vol = unit_len ** 3

class PwscfKS(Driver):
    """
    Note :
        The extpot separated into two parts : v.of_r and vltot will be a better and safe way
    """
    def __init__(self, evaluator = None, subcell = None, prefix = 'sub_ks', params = None, cell_params = None,
            exttype = 3, base_in_file = None, mixer = None, ncharge = None, options = None, comm = None, **kwargs):
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
        super().__init__(options = options, technique = 'KS', **kwargs)
        self._input_ext = '.in'

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
            self.prefix, self._input_ext= os.path.splitext(base_in_file)

        if self.comm.size > 1 : self.comm.Barrier()
        self.mixer = mixer
        self._grid = None
        self._grid_sub = None
        self._driver_initialise()
        self.grid_driver = self.get_grid_driver(self.grid)
        #-----------------------------------------------------------------------
        self.nspin = 1
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
            fstr += f'{self.__class__.__name__} has two grids :{self.grid.nrR} and {self.grid_driver.nrR}'
        sprint(fstr, comm=self.comm, level=1)
        pwscfpy.pwpy_mod.pwpy_write_stdout(fstr)
        #-----------------------------------------------------------------------
        self.init_density()
        self.update_workspace(first = True)
        self.embed = pwscfpy.pwpy_embed.embed_base()

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
        if isinstance(self.mixer, AbstractMixer):
            self.mixer.restart()
        if subcell is not None :
            self.subcell = subcell
        self.gaussian_density = self.get_gaussian_density(self.subcell, grid = self.grid)

        if not first :
            pos = self.subcell.ions.pos.to_crys().T
            pwscfpy.pwpy_mod.pwpy_update_ions(self.embed, pos)
            # get new density
            pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
            self.density[:] = self._format_density_invert()

        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        if self.comm.rank == 0 :
            core_charge = np.empty((grid.nnr, self.nspin), order = 'F')
        else :
            core_charge = self.atmp2
        pwscfpy.pwpy_mod.pwpy_get_rho_core(core_charge)
        self.core_density= self._format_density_invert(core_charge)

        self.core_density_sub = Field(grid = self.grid_sub)
        self.grid_sub.scatter(self.core_density, out = self.core_density_sub)

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
            mp = MP(comm = self.comm)
            self._grid_sub = Grid(self.subcell.grid.lattice, self.subcell.grid.nrR, direct = True, mp = mp)
        return self._grid_sub

    def get_grid_driver(self, grid):
        nr = np.zeros(3, dtype = 'int32')
        pwscfpy.pwpy_mod.pwpy_get_grid(nr)
        if not np.all(grid.nrR == nr) and self.comm.rank == 0 :
            grid_driver = Grid(grid.lattice, nr, direct = True)
        else :
            grid_driver = None
        return grid_driver

    def build_input(self, params, cell_params, base_in_file):
        in_params, cards = self._build_ase_atoms(params, cell_params, base_in_file)
        self._write_params(self.prefix + self._input_ext, params = in_params, cell_params = cell_params, cards = cards)

    def _fix_params(self, params = None):
        prefix = self.prefix
        #output of qe
        self.outfile = prefix + '.out'
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

    def _build_ase_atoms(self, params = None, cell_params = None, base_in_file = None):
        ase_atoms = ions2ase(self.subcell.ions)
        ase_atoms.set_calculator(ase_calc_driver())
        self.ase_atoms = ase_atoms

        if base_in_file :
            fileobj = open(base_in_file, 'r')
            in_params, card_lines = ase_io_driver.read_fortran_namelist(fileobj)
            fileobj.close()
        else :
            in_params = {}
            card_lines = []

        in_params = self._fix_params(in_params)

        for k1, v1 in params.items() :
            if k1 not in in_params :
                in_params[k1] = {}
            for k2, v2 in v1.items() :
                in_params[k1][k2] = v2

        return in_params, card_lines

    def _driver_initialise(self, **kwargs):
        if self.comm is None or isinstance(self.comm, SerialComm):
            comm = None
        else :
            comm = self.comm.py2f()
            # print('comm00', comm, self.comm.size)
        pwscfpy.pwpy_mod.pwpy_set_stdout(self.outfile)
        pwscfpy.pwpy_pwscf(self.prefix + self._input_ext, comm)

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

        if rho_ini is not None :
            self.density[:] = rho_ini
            self._format_density()
        else :
            pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
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
        fileobj = open(outfile, 'w')
        if 'kpts' not in cell_params :
            self._update_kpoints(cell_params, cards)
        ase_io_driver.write_espresso_in(fileobj, self.ase_atoms, params, **cell_params)
        self._write_params_cards(fileobj, params, cards)
        fileobj.close()
        return

    def _update_kpoints(self, cell_params, cards = None):
        if cards is None or len(cards) == 0 :
            return
        lines = iter(cards)
        for line in lines :
            if line.split()[0].upper() == 'K_POINTS' :
                ktype = line.split()[1].lower()
                if ktype == 'gamma' :
                    break
                elif ktype == 'automatic' :
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

    def _format_density(self, volume = None, sym = False, **kwargs):
        self.prev_charge, self.charge = self.charge, self.prev_charge
        if self.grid_driver is not None and self.comm.rank == 0:
            charge = grid_map_data(self.density, grid = self.grid_driver)
        else :
            charge = self.density
        #-----------------------------------------------------------------------
        charge = charge.reshape((-1, self.nspin), order='F') / unit_vol
        self.charge[:] = charge
        pwscfpy.pwpy_mod.pwpy_set_rho(charge)
        # if sym :
            # pwscfpy.pwpy_sum_band_sym()
            # pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
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

    def get_density(self, vext = None, sdft = 'sdft', **kwargs):
        '''
        '''
        #-----------------------------------------------------------------------
        printout = 2
        exxen = 0.0
        #-----------------------------------------------------------------------
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

        # if self.comm.rank == 0:
        #     from dftpy.formats import io
        #     io.write(self.prefix + '.xsf', self.evaluator.embed_potential, self.subcell.ions)

        self.prev_charge[:] = self.charge

        pwscfpy.pwpy_mod.pwpy_set_extpot(self.embed, extpot)
        self.embed.exttype = self.exttype
        self.embed.initial = initial
        self.embed.mix_coef = -1.0
        self.embed.finish = False
        pwscfpy.pwpy_electrons.pwpy_electrons_scf(printout, exxen, self.embed)
        self.energy = self.embed.etotal
        self.dp_norm = self.embed.dnorm

        pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
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
            pwscfpy.pwpy_mod.pwpy_set_extpot(self.embed, extpot)
            pwscfpy.pwpy_calc_energies(self.embed)
        energy = self.embed.etotal * 0.5
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], olevel = 1, sdft = 'sdft', **kwargs):
        if 'E' in calcType :
            energy = self.get_energy(olevel = olevel, sdft = sdft)
            if sdft == 'pdft' :
                func = Functional(name = 'ZERO', energy=0.0, potential=None)
            else :
                if olevel == 0 :
                    self.grid_sub.scatter(density, out = self.density_sub)
                    func = self.evaluator(self.density_sub, calcType = ['E'], with_global = False, with_embed = False, gather = True)
                else : # elif olevel == 1 :
                    if self.comm.rank == 0 :
                        func = self.evaluator(density, calcType = ['E'], with_global = False, with_embed = True)
                    else :
                        func = Functional(name = 'ZERO', energy=0.0, potential=None)

            func.energy += energy
            if sdft == 'sdft' and self.exttype == 0 : func.energy = 0.0
            if self.comm.rank > 0 : func.energy = 0.0

            fstr = f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm, level=1)
            pwscfpy.pwpy_mod.pwpy_write_stdout(fstr)
            self.energy = func.energy
        return func

    def update_density(self, coef = None, **kwargs):
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
            self.dp_norm = hartree_energy(r)
            rmax = r.amax()
            fstr = f'res_norm({self.prefix}): {self._iter}  {rmax}  {self.residual_norm}'
            sprint(fstr, comm=self.comm, level=1)
            pwscfpy.pwpy_mod.pwpy_write_stdout(fstr)
            #-----------------------------------------------------------------------
            if self.mix_driver is None :
                rho = self.mixer(prev_density, density, **kwargs)
                if self.grid_driver is not None and mix_grid:
                    rho = grid_map_data(rho, grid = self.grid)
                self.density[:] = rho

        if self.mix_driver is not None :
            if coef is None : coef = self.mix_driver
            self.embed.mix_coef = coef
            pwscfpy.pwpy_electrons.pwpy_electrons_scf(0, 0, self.embed)
            if self._iter > 1 : self.dp_norm = self.embed.dnorm
            pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
            self.density[:] = self._format_density_invert()

        if self.comm.rank > 0 :
            self.residual_norm = 0.0
            self.dp_norm = 0.0
        # if self.comm.size > 1 : self.residual_norm = self.comm.bcast(self.residual_norm, root=0)
        return self.density

    def get_fermi_level(self, **kwargs):
        results = pwscfpy.ener.get_ef()
        return results

    def get_forces(self, icalc = 2, **kwargs):
        """
        icalc :
                0 : all
                1 : no ewald
                2 : no ewald and local_potential
        """
        pwscfpy.pwpy_forces(icalc)
        # forces = pwscfpy.force_mod.force.T
        forces = pwscfpy.force_mod.get_array_force().T / 2.0  # Ry to a.u.
        return forces

    def get_stress(self, **kwargs):
        pass

    def end_scf(self, **kwargs):
        self.embed.finish = True
        pwscfpy.pwpy_electrons.pwpy_electrons_scf(0, 0, self.embed)

    def stop_run(self, status = 0, **kwargs):
        pwscfpy.pwpy_stop_run(status, **kwargs)
