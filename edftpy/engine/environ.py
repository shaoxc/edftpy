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
from edftpy.engine.driver import DriverEX


class DriverEnviron(DriverEX):
    """
    Note :
        The potential and density will gather in rank == 0 for engine.
    """
    def __init__(self, **kwargs):
        kwargs["technique"] = 'EN'
        super().__init__(**kwargs)

    def get_energy_potential(self, density = None, calcType = ['E', 'V'], olevel = 1, **kwargs):
        func = Functional(name = 'ZERO', energy=0.0, potential=None)
        self.engine.set_extpot(self.evaluator.global_potential, **kwargs)
        self.engine.scf(density, **kwargs)
        if 'V' in calcType :
            pot = self.engine.get_potential(**kwargs)
            func.potential = self._format_field(pot)
        if 'E' in calcType :
            energy = self.get_energy(olevel = olevel)
            func.energy = energy
            if self.comm.rank > 0 : func.energy = 0.0
            fstr = f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm, level=1)
            self.engine.write_stdout(fstr)
        return func
