import os
import numpy as np
from dftpy.constants import ENERGY_CONV, FORCE_CONV, STRESS_CONV
import ase

from edftpy.io import ase2ions
from edftpy.interface import config2optimizer
from edftpy.mpi import sprint

class eDFTpyCalculator(object):
    """eDFTpy calculator for ase"""
    def __init__(self, config=None, graphtopo = None, atoms = None):
        self.config = config
        self.graphtopo = graphtopo
        self.optimizer = None
        self.restart()
        self.iter = 0
        if atoms is None :
            cell_file = config["PATH"]["cell"] +os.sep+ config['GSYSTEM']["cell"]["file"]
            atoms = ase.io.read(cell_file)
        self.atoms = atoms
        self.atoms.calc = self
        self.atoms_save = None

    def restart(self):
        self._energy = None
        self._forces = None
        self._stress = None

    def check_restart(self, atoms=None):
        self.atoms = atoms
        if (self.atoms_save and atoms == self.atoms_save):
            return False
        else:
            self.atoms_save = atoms.copy()
            self.restart()
            return True

    def update_optimizer(self, atoms = None):
        atoms = atoms or self.atoms
        ions = ase2ions(atoms)
        if self.iter > 1 :
            append = True
        else :
            append = False
        self.optimizer = config2optimizer(self.config, ions, self.optimizer, graphtopo = self.graphtopo, append = append)
        self.optimizer.optimize()
        self.iter += 1

    def get_potential_energy(self, atoms, olevel = None, **kwargs):
        if self.check_restart(atoms):
            self.update_optimizer(atoms)
        if olevel is not None :
            if olevel == 0 :
                self.optimizer.energy = self.optimizer.print_energy()['TOTAL']
            else :
                self.optimizer.energy = self.optimizer.get_energy(olevel = olevel)
        self._energy = self.optimizer.energy
        sprint('Total energy :', self._energy * ENERGY_CONV["Hartree"]["eV"], self.graphtopo.size, self.iter)
        return self._energy * ENERGY_CONV["Hartree"]["eV"]

    def get_forces(self, atoms):
        if self.check_restart(atoms):
            self.update_optimizer(atoms)
        if self._forces is None :
            self._forces = self.optimizer.get_forces()
        return self._forces * FORCE_CONV["Ha/Bohr"]["eV/A"]

    def get_stress(self, atoms):
        if self.check_restart(atoms):
            self.update_optimizer(atoms)
        stress_voigt = np.zeros(6)
        return stress_voigt * STRESS_CONV["Ha/Bohr3"]["eV/A3"]

    def output_density(self, **kwargs):
        return self.optimizer.output_density(**kwargs)

    def _add_calc_no(cls, dftd4 = None, config=None, graphtopo = None, **kwargs):
        obj = super(eDFTpyCalculator, cls).__new__(cls)
        if dftd4 :
            from .dftd4 import VDWDFTD4
            from ase.calculators.mixing import SumCalculator
            xc_options = config['GSYSTEM']['exc'].copy()
            xc_options.pop('dftd4', None)
            xc_options.update(kwargs)
            calc = VDWDFTD4(dftd4 = dftd4, **xc_options).dftd4calculator
            obj = SumCalculator([obj, calc])
        return obj
