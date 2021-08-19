import numpy as np
from scipy import signal
from abc import ABC, abstractmethod
import os

from dftpy.constants import ENERGY_CONV

from edftpy.mixer import PulayMixer, AbstractMixer
from edftpy.utils.common import Grid, Field, Functional
from edftpy.utils.math import grid_map_data
from edftpy.utils import clean_variables
from edftpy.mpi import sprint, SerialComm, MP
from edftpy.engine.driver import Driver


class DriverEX(Driver):
    """
    Note :
        The potential and density will gather in rank == 0 for engine.
    """
    def __init__(self, engine = None, **kwargs):
        kwargs["technique"] = 'EX'
        super().__init__(**kwargs)
        self.engine = engine

        self.prefix, self._input_ext= os.path.splitext(self.base_in_file)

        if self.comm.size > 1 : self.comm.Barrier()
        self._grid = None
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
            fstr += f'{self.__class__.__name__} has two grids :{self.grid.nrR} and {self.grid_driver.nrR}'
        sprint(fstr, comm=self.comm, level=1)
        self.engine.write_stdout(fstr)
        #-----------------------------------------------------------------------
        self.init_density()
        self.embed = self.engine.embed_base(**kwargs)
        self.update_workspace(first = True, restart = self.restart)

    def update_workspace(self, subcell = None, first = False, update = 0, restart = False, **kwargs):
        """
        Notes:
            clean workspace
        """
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
        self.engine.get_grid(nr)
        if not np.all(grid.nrR == nr) and self.comm.rank == 0 :
            grid_driver = Grid(grid.lattice, nr, direct = True)
        else :
            grid_driver = None
        return grid_driver

    def _driver_initialise(self, append = False, **kwargs):
        if self.comm.rank == 0 :
            self.engine.set_stdout(self.outfile, append = append)
        if self.task == 'optical' :
            self.engine.tddft_initial(self.prefix + self._input_ext, self.comm)
        else :
            self.engine.initial(self.prefix + self._input_ext, self.comm)

    def _format_field(self, density, **kwargs):
        if self.grid_driver is not None and self.comm.rank == 0:
            charge = grid_map_data(density, grid = self.grid_driver)
        else :
            charge = density
        #-----------------------------------------------------------------------
        charge = charge.reshape((-1, self.nspin), order=self.engine.units['order']) / self.engine.units['volume']
        return

    def _format_field_invert(self, charge, grid = None, **kwargs):
        if grid is None :
            grid = self.grid

        if self.grid_driver is not None and np.any(self.grid_driver.nrR != grid.nrR):
            density = Field(grid=self.grid_driver, direct=True, data = charge, order = self.engine.units['order'])
            rho = grid_map_data(density, grid = grid)
        else :
            rho = Field(grid=grid, rank=1, direct=True, data = charge, order = self.engine.units['order'])
        rho *= self.engine.units['volume']
        return rho

    def get_energy(self, olevel = 0, **kwargs):
        if olevel == 0 :
            energy = self.engine.calc_energy(self.embed)
        else :
            energy = 0.0
        return energy

    def get_energy_potential(self, density = None, calcType = ['E', 'V'], olevel = 1, **kwargs):
        func = Functional(name = 'ZERO', energy=0.0, potential=None)
        if 'E' in calcType :
            energy = self.get_energy(olevel = olevel)
            func.energy = energy
            if self.comm.rank > 0 : func.energy = 0.0
            fstr = f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm, level=1)
            self.engine.write_stdout(fstr)
        if 'V' in calcType :
            pot = self.engine.get_potential(**kwargs)
            func.potential = self._format_field(pot)
        return func
