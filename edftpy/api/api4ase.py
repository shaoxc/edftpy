import numpy as np
from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.formats.ase_io import ase2ions

from edftpy.utils.common import Field, Grid, Atoms, Coord
from edftpy.interface import config2optimizer, get_forces


class eDFTpyCalculator(object):
    """eDFTpy calculator for ase"""

    def __init__(self, config=None):
        self.config = config
        self.atoms = None
        self.optimizer = None
        self.restart()

    def restart(self):
        self._energy = None
        self._forces = None
        self._stress = None

    def check_restart(self, atoms=None):
        if (self.atoms and atoms == self.atoms):
            return False
        else:
            self.atoms = atoms.copy()
            self.restart()
            return True

    def update_optimizer(self, atoms = None):
        ions = ase2ions(atoms)
        self.optimizer = config2optimizer(self.config, ions, self.optimizer)
        self.optimizer.optimize()

    def get_potential_energy(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
            self.update_optimizer(atoms)
        self._energy = self.optimizer.energy
        print('Total energy :', self._energy * ENERGY_CONV["Hartree"]["eV"])
        return self._energy * ENERGY_CONV["Hartree"]["eV"]

    def get_forces(self, atoms):
        if self.check_restart(atoms):
            self.update_optimizer(atoms)
        if self._forces is None :
            self._forces = get_forces(self.optimizer.opt_drivers, self.optimizer.gsystem)
        return self._forces * FORCE_CONV["Ha/Bohr"]["eV/A"]

    def get_stress(self, atoms):
        pass
