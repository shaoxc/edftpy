import numpy as np
from scipy import signal
import copy
import os
from dftpy.formats.ase_io import ions2ase
import pwscfpy
import ase.io.espresso as ase_io_driver
from ase.calculators.espresso import Espresso as ase_calc_driver

from ..mixer import LinearMixer, PulayMixer
from ..utils.common import AbsDFT, Grid, Field
from ..utils.math import grid_map_data
from ..density import normalization_density

class PwscfKS(AbsDFT):
    """description"""
    def __init__(self, evaluator = None, subcell = None, prefix = 'qe_sub_in', params = None, cell_params = None, 
            exttype = 3, base_in_file = None, mixer = None, ncharge = None, **kwargs):
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
        self.evaluator = evaluator
        self.prefix = prefix + '.in'
        self.exttype = exttype
        self.subcell = subcell
        self.rho = None
        self.wfs = None
        self.occupations = None
        self.eigs = None
        self.fermi = None
        self.perform_mix = False
        self.ncharge = ncharge
        if self.prefix :
            in_params = self._build_ase_atoms(params, cell_params, base_in_file)
            self._write_params(self.prefix, params = in_params, cell_params = cell_params)
        else :
            self.prefix = base_in_file

        self._driver_initialise(self.prefix)
        self._iter = 0
        self._filter = None
        self.mixer = mixer

        if 'system' in params :
            fixnr = params['system'].get('nr1', None)
        if fixnr :
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
        nr = np.zeros(3, dtype = 'int32')
        pwscfpy.pwpy_mod.pwpy_get_grid(nr)
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

        return in_params

    def _driver_initialise(self, infile = None, **kwargs):
        pwscfpy.pwpy_pwscf(infile)

    def init_density(self, rho_ini = None):
        if rho_ini is not None :
            self._format_density(rho_ini, sym = False)

        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid
        self.charge = np.empty((grid.nnr, 1), order = 'F')
        pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)

        density = self._format_density_invert(self.charge, self.grid)
        density = normalization_density(density, ncharge = self.ncharge, grid = self.grid)
        self._format_density(density, sym = False)

    def _write_params(self, outfile, params = None, cell_params = None, **kwargs):
        cell_params.update(kwargs)
        if 'pseudopotentials' not in cell_params :
            raise AttributeError("!!!ERROR : Must give the pseudopotentials")
        value = {}
        for k2, v2 in cell_params['pseudopotentials'].items():
            value[k2] = os.path.basename(v2)
        cell_params['pseudopotentials'].update(value)
        #For QE, all pseudopotentials should at same directory
        params['control']['pseudo_dir'] = os.path.dirname(v2)

        fileobj = open(outfile, 'w')
        ase_io_driver.write_espresso_in(fileobj, self.ase_atoms, params, **cell_params)
        fileobj.close()
        return

    def _format_density(self, density, volume = None, sym = True, **kwargs):
        #-----------------------------------------------------------------------
        if volume is None :
            volume = density.grid.volume
        if self.grid_driver is not None :
            charge = grid_map_data(density, grid = self.grid_driver)
        else :
            charge = density 
        # charge *= volume
        #-----------------------------------------------------------------------
        charge = charge.reshape((-1, 1), order='F')
        pwscfpy.pwpy_mod.pwpy_set_rho(charge)
        if sym :
            pwscfpy.pwpy_sum_band_sym()
        pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
        return

    def _format_density_invert(self, charge = None, grid = None, **kwargs):
        if charge is None :
            charge = self.charge

        if grid is None :
            grid = self.grid

        density = charge.copy()

        if self.grid_driver is not None and np.any(self.grid_driver.nr != grid.nr):
            # density /= grid.volume
            density = Field(grid=self.grid_driver, direct=True, data = density, order = 'F')
            rho = grid_map_data(density, grid = grid)
        else :
            # density /= grid.volume
            rho = Field(grid=grid, rank=1, direct=True, data = density, order = 'F')
        return rho

    def _get_extpot(self, charge = None, grid = None, **kwargs):
        rho = self._format_density_invert(charge, grid, **kwargs)
        # func = self.evaluator(rho, with_embed = False)
        # # func.potential *= self.filter
        # extpot = func.potential.ravel(order = 'F')
        # extene = func.energy
        #-----------------------------------------------------------------------
        self.evaluator.get_embed_potential(rho, gaussian_density = self.subcell.gaussian_density, with_global = True)
        extpot = self.evaluator.embed_potential
        extene = (extpot * rho).integral()
        if self.grid_driver is not None :
            extpot = grid_map_data(extpot, grid = self.grid_driver)
        extpot = extpot.ravel(order = 'F') * 2.0 # a.u. to Ry
        #-----------------------------------------------------------------------
        return extpot, extene

    def get_density(self, density, vext = None, **kwargs):
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

        self._format_density(density, sym = sym)
        extpot, extene = self._get_extpot(self.charge, density.grid)

        self.prev_charge = self.charge.copy()

        if self._iter > 0 :
            pwscfpy.pwpy_electrons_scf(printout, exxen, extpot, extene, self.exttype, initial)
        else :
            pwscfpy.pwpy_electrons_scf(printout, exxen, extpot, extene, 0, initial)

        pwscfpy.pwpy_sum_band()
        if sym :
            pwscfpy.pwpy_sum_band_sym()
        pwscfpy.pwpy_mod.pwpy_get_rho(self.charge)
        rho = self._format_density_invert(self.charge, self.grid)
        # print('KS_Chemical_potential ', self.get_fermi_level())
        return rho

    def get_energy(self, density = None, **kwargs):
        extpot = np.zeros(self.charge.size)
        energy = pwscfpy.pwpy_calc_energies(extpot, 0.0, self.exttype) * 0.5
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], **kwargs):
        func = self.evaluator(density, calcType = ['E'], with_global = False, with_embed = False)
        if self.exttype == 0 :
            func.energy = 0.0
        if 'E' in calcType :
            func.energy += self.get_energy()
        print('sub_energy_ks', func.energy)
        return func

    def update_density(self, **kwargs):
        mix_grid = False
        # mix_grid = True
        if self.grid_driver is not None and mix_grid:
            grid = self.grid_driver
        else :
            grid = self.grid
        prev_density = self._format_density_invert(self.prev_charge, grid)
        density = self._format_density_invert(self.charge, grid)
        #-----------------------------------------------------------------------
        r = density - prev_density
        print('res_norm_ks', self._iter, np.max(abs(r)), np.sqrt(np.sum(r * r)/np.size(r)))
        #-----------------------------------------------------------------------
        rho = self.mixer(prev_density, density, **kwargs)
        if self.grid_driver is not None and mix_grid:
            rho = grid_map_data(rho, grid = self.grid)
        return rho

    def get_fermi_level(self, **kwargs):
        results = pwscfpy.ener.ef
        return results

    def get_forces(self, icalc = 2, **kwargs):
        """
        icalc :
                0 : all
                1 : no ewald
                2 : no ewald and local_potential
        """
        pwscfpy.pwpy_forces(icalc)
        forces = pwscfpy.force_mod.force.T
        return forces

    def get_stress(self, **kwargs):
        pass
