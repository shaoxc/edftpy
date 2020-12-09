import numpy as np
from scipy import signal
import copy
import os
from dftpy.formats.ase_io import ions2ase
import pwscfpy
import ase.io.espresso as ase_io_driver
from ase.calculators.espresso import Espresso as ase_calc_driver

from ..mixer import LinearMixer, PulayMixer
from ..utils.common import Grid, Field, Functional
from ..utils.math import grid_map_data
from ..density import normalization_density
from .driver import Driver
from edftpy.mpi import sprint, SerialComm

class PwscfKS(Driver):
    """description"""
    def __init__(self, evaluator = None, subcell = None, prefix = 'qe_sub_in', params = None, cell_params = None,
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
        Driver.__init__(self, options = options)
        self.evaluator = evaluator
        self.exttype = exttype
        self.subcell = subcell
        self.prefix = prefix
        self.rho = None
        self.wfs = None
        self.occupations = None
        self.eigs = None
        self.fermi = None
        self.perform_mix = False
        self.ncharge = ncharge
        self.comm = self.subcell.grid.mp.comm
        self.outfile = None
        if self.prefix :
            self.prefix += '.in'
            if self.comm.rank == 0 :
                self.build_input(params, cell_params, base_in_file)
        else :
            self.prefix = base_in_file

        if self.comm.size > 1 : self.comm.Barrier()
        self._driver_initialise()
        self._iter = 0
        self._filter = None
        self.mixer = mixer

        self._grid = None
        self.grid_driver = self.get_grid_driver(self.grid)
        #-----------------------------------------------------------------------
        self.init_density()
        #-----------------------------------------------------------------------
        self.mix_driver = None
        if self.mixer is None :
            self.mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.1], maxm = 7, coef = [0.5], predecut = 0, delay = 1)
            # rho0 = np.mean(self.density)
            # kf = (3.0 * rho0 * np.pi ** 2) ** (1.0 / 3.0)
            # self.mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, kf, 1.0], maxm = 7, coef = [0.7], predecut = 0, delay = 1)
        elif isinstance(self.mixer, float):
            self.mix_driver = self.mixer
        if self.grid_driver is not None :
            sprint('{} has two grids :{} and {}'.format(self.__class__.__name__, self.grid.nr, self.grid_driver.nr), comm=self.comm)
        #-----------------------------------------------------------------------
        self.gaussian_density = self.get_gaussian_density(self.subcell, grid = self.grid)
        self.energy = 0.0

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
        self._write_params(self.prefix, params = in_params, cell_params = cell_params, cards = cards)

    def _build_ase_atoms(self, params = None, cell_params = None, base_in_file = None):
        ase_atoms = ions2ase(self.subcell.ions)
        ase_atoms.set_calculator(ase_calc_driver())
        self.ase_atoms = ase_atoms

        if base_in_file is not None :
            fileobj = open(base_in_file, 'r')
            in_params, card_lines = ase_io_driver.read_fortran_namelist(fileobj)
            fileobj.close()
        else :
            in_params = {}

        fix_params = {'electrons' : {'electron_maxstep' : 1}}
        if params is None :
            params = {}
        params.update(fix_params)

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
        pwscfpy.pwpy_pwscf(self.prefix, comm)

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
        # outdir of qe
        prefix = os.path.splitext(self.prefix)[0]
        params['control']['prefix'] = prefix
        params['control']['outdir'] = prefix + '.tmp'
        # do not save anything for qe
        params['control']['disk_io'] = 'none'
        ase_io_driver.write_espresso_in(fileobj, self.ase_atoms, params, **cell_params)
        self._write_params_cards(fileobj, params, cards)
        fileobj.close()
        #output of qe
        self.outfile = prefix + '.out'
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
        charge = charge.reshape((-1, 1), order='F')
        pwscfpy.pwpy_mod.pwpy_set_rho(charge)
        if sym :
            pwscfpy.pwpy_sum_band_sym()
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
        extpot = extpot.ravel(order = 'F') * 2.0 # a.u. to Ry
        extene *= -2.0
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

        # extene = 0.0
        if self._iter > 0 :
        # if self._iter > 100 :
            self.energy = pwscfpy.pwpy_electrons_scf(printout, exxen, extpot, extene, self.exttype, initial, self.mix_driver)
        else :
            self.energy = pwscfpy.pwpy_electrons_scf(printout, exxen, extpot, extene, 0, initial, self.mix_driver)

        if self.mix_driver is None :
            pwscfpy.pwpy_sum_band()
            if sym :
                pwscfpy.pwpy_sum_band_sym()
        pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
        self.density[:] = self._format_density_invert()
        return self.density

    def get_energy(self, **kwargs):
        energy = pwscfpy.pwpy_calc_energies(self.exttype) * 0.5
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], **kwargs):
        if 'E' in calcType :
            energy = self.get_energy()
            # energy = self.energy

        if self.comm.rank == 0 :
            func = self.evaluator(density, calcType = ['E'], with_global = False, with_embed = False)
            if self.exttype == 0 :
                func.energy = 0.0
        else :
            func = Functional(name = 'ZERO', energy=0.0, potential=None)
            energy = 0.0

        if 'E' in calcType :
            func.energy += energy
            fstr = f'sub_energy_ks : {self._iter}, {func.energy}'
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
            rmax = r.amax()
            fstr = f'res_norm_ks : {self._iter}, {rmax}, {self.residual_norm}'
            sprint(fstr, comm=self.comm)
            pwscfpy.pwpy_mod.pwpy_write_stdout(fstr)
            #-----------------------------------------------------------------------
            if self.mix_driver is not None :
                pass
            else :
                rho = self.mixer(prev_density, density, **kwargs)
                if self.grid_driver is not None and mix_grid:
                    rho = grid_map_data(rho, grid = self.grid)
                self.density[:] = rho
        else :
            self.residual_norm = 100.0
        if self.comm.size > 1 :
            self.residual_norm = self.comm.bcast(self.residual_norm, root=0)
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
