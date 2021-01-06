import pwscfpy

import numpy as np
from scipy import signal
import copy
import os
from dftpy.formats.ase_io import ions2ase
from dftpy.constants import LEN_CONV, ENERGY_CONV
import ase.io.espresso as ase_io_driver
from ase.calculators.espresso import Espresso as ase_calc_driver

from ..mixer import LinearMixer, PulayMixer, AbstractMixer
from ..utils.common import Grid, Field, Functional
from ..utils.math import grid_map_data
from ..density import normalization_density
from .driver import Driver
from edftpy.hartree import hartree_energy
from edftpy.mpi import sprint, SerialComm
from collections import OrderedDict

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
        super().__init__(options = options, technique = 'KS')
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
        self._driver_initialise()
        self.mixer = mixer

        self._grid = None
        self.grid_driver = self.get_grid_driver(self.grid)
        #-----------------------------------------------------------------------
        self.init_density()
        #-----------------------------------------------------------------------
        self.mix_driver = None
        if self.mixer is None :
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, 0.6, 1.0], maxm = 7, coef = [0.5], predecut = 0, delay = 1)
        elif isinstance(self.mixer, float):
            self.mix_driver = self.mixer
        if self.grid_driver is not None :
            sprint('{} has two grids :{} and {}'.format(self.__class__.__name__, self.grid.nr, self.grid_driver.nr), comm=self.comm)
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
        self.dp_norm = 1
        if isinstance(self.mixer, AbstractMixer):
            self.mixer.restart()
        if subcell is not None :
            self.subcell = subcell
        self.gaussian_density = self.get_gaussian_density(self.subcell, grid = self.grid)
        if not first :
            # from qe-6.5:run_pwscf.f90
            pwscfpy.pwpy_electrons_scf(0, 0, self.charge, 0, self.exttype, 0, finish = True)
            pwscfpy.extrapolation.update_pot()
            pwscfpy.hinit1()

        # self.core_charge = pwscfpy.scf.get_array_rho_core()
        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid
        self.core_charge = np.empty((grid.nnr, 1), order = 'F')
        pwscfpy.pwpy_mod.pwpy_get_rho_core(self.core_charge)
        if self.comm.rank == 0 :
            self.core_density = self._format_density_invert(self.core_charge)
        else :
            self.core_density = Field(grid=self.grid, rank=1, direct=True)
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
        nr = np.zeros(3, dtype = 'int32')
        pwscfpy.pwpy_mod.pwpy_get_grid(nr)
        if not np.all(grid.nrR == nr):
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
                    'verbosity' : 'high',
                    'restart_mode' : 'from_scratch',
                    'iprint' : 1,
                    },
                'system' :
                {
                    'ibrav' : 0,
                    'nat' : 1,
                    'ntyp' : 1,
                    # 'ecutwfc' : 40,
                    'nosym' : True,
                    'occupations' : 'smearing',
                    'degauss' : 0.001,
                    'smearing' : 'gaussian',
                    },
                'electrons' :
                {
                    'electron_maxstep' : 1,
                    'conv_thr' : 1E-8,
                    'mixing_beta' : 0.5,
                    },
                'ions' :
                {
                    'pot_extrapolation' : 'none',
                    },
                'cell' :{}
                })
        fix_params = {
                'control' :
                {
                    'disk_io' : 'none', # do not save anything for qe
                    'prefix' : prefix,
                    'outdir' : prefix + '.tmp'  # outdir of qe
                    },
                'electrons' :
                {
                    'electron_maxstep' : 1  # only run 1 step for scf
                    },
                'ions' :
                {
                    'pot_extrapolation' : 'none',  # no extrapolation for potential from preceding ionic steps
                    'wfc_extrapolation' : 'none'   # no extrapolation for wfc from preceding ionic steps
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
        self.density = Field(grid=self.grid)
        self.prev_density = Field(grid=self.grid)
        if rho_ini is not None :
            self._format_density(rho_ini, sym = False)

        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        self.charge = np.empty((grid.nnr, 1), order = 'F')
        self.prev_charge = np.empty((grid.nnr, 1), order = 'F')
        pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)

        if self.comm.rank == 0 :
            self.density[:] = self._format_density_invert()
            # print('ncharge', self.density.integral())
            self.density = normalization_density(self.density, ncharge = self.ncharge, grid = self.grid)
        self._format_density(sym = False)

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

    def _format_density(self, volume = None, sym = True, **kwargs):
        #-----------------------------------------------------------------------
        # self.prev_density[:] = self.density
        self.prev_charge, self.charge = self.charge, self.prev_charge
        if self.grid_driver is not None :
            charge = grid_map_data(self.density, grid = self.grid_driver)
        else :
            charge = self.density
        #-----------------------------------------------------------------------
        charge = charge.reshape((-1, 1), order='F') / unit_vol
        pwscfpy.pwpy_mod.pwpy_set_rho(charge)
        # if sym :
            # pwscfpy.pwpy_sum_band_sym()
        pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
        #-----------------------------------------------------------------------
        self.density[:] = self._format_density_invert(**kwargs)
        self.prev_density[:] = self.density
        return

    def _format_density_invert(self, charge = None, grid = None, **kwargs):
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

    def _get_extpot(self, **kwargs):
        if self.comm.rank == 0 :
            self.evaluator.get_embed_potential(self.density, gaussian_density = self.gaussian_density)
            extpot = self.evaluator.embed_potential
            extene = (extpot * self.density).integral()
            if self.grid_driver is not None :
                extpot = grid_map_data(extpot, grid = self.grid_driver)
        else :
            extene = 0.0
            if self.grid_driver is not None :
                extpot = np.empty(self.grid_driver.nrR, order = 'F')
            else :
                extpot = np.empty(self.grid.nrR, order = 'F')
        if self.comm.size > 1 :
            extene = self.comm.bcast(extene, root = 0)
        extpot = extpot.ravel(order = 'F') * 2.0 # a.u. to Ry
        # extene *= -2.0
        extene = 0.0
        return extpot, extene

    def get_density(self, vext = None, **kwargs):
        '''
        Must call first time
        '''
        #-----------------------------------------------------------------------
        printout = 2
        exxen = 0.0
        #-----------------------------------------------------------------------
        self._iter += 1
        sym = True
        if self._iter == 1 :
            initial = True
        else :
            initial = False

        self._format_density(sym = sym)
        extpot, extene = self._get_extpot()

        self.prev_charge[:] = self.charge

        self.energy, self.dp_norm = pwscfpy.pwpy_electrons_scf(printout, exxen, extpot, extene, self.exttype, initial)

        # if self.mix_driver is None :
            # pwscfpy.pwpy_sum_band()
            # if sym :
                # pwscfpy.pwpy_sum_band_sym()

        pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
        self.density[:] = self._format_density_invert()
        return self.density

    def get_energy(self, **kwargs):
        energy = pwscfpy.pwpy_calc_energies(self.exttype) * 0.5
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], olevel = 1, **kwargs):
        # olevel =0
        if 'E' in calcType :
            if olevel == 0 :
                energy = self.get_energy()
            else : # elif olevel == 1 :
                energy = self.energy

        if self.comm.rank == 0 :
            func = self.evaluator(density, calcType = ['E'], with_global = False, with_embed = False)
            if self.exttype == 0 :
                func.energy = 0.0
        else :
            func = Functional(name = 'ZERO', energy=0.0, potential=None)
            energy = 0.0

        if 'E' in calcType :
            func.energy += energy
            fstr = f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm)
            pwscfpy.pwpy_mod.pwpy_write_stdout(fstr)
            self.energy = func.energy
        return func

    def update_density(self, **kwargs):
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
            sprint(fstr, comm=self.comm)
            pwscfpy.pwpy_mod.pwpy_write_stdout(fstr)
            #-----------------------------------------------------------------------
            if self.mix_driver is None :
                rho = self.mixer(prev_density, density, **kwargs)
                if self.grid_driver is not None and mix_grid:
                    rho = grid_map_data(rho, grid = self.grid)
                self.density[:] = rho
        else :
            self.residual_norm = 0.0
            self.dp_norm = 0.0

        if self.mix_driver is not None :
            ene, dp_norm = pwscfpy.pwpy_electrons_scf(0, 0, self.charge[:, 0], 0, self.exttype, 0, mix_coef = self.mix_driver)
            if self._iter > 1 : self.dp_norm = dp_norm
            pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
            self.density[:] = self._format_density_invert()

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
        forces = pwscfpy.force_mod.get_array_force().T
        return forces

    def get_stress(self, **kwargs):
        pass

    def stop_run(self, *arg, **kwargs):
        pwscfpy.pwpy_stop_run(*arg, **kwargs)
