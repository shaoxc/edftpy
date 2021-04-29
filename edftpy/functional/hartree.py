from edftpy.utils.common import AbsFunctional
from dftpy.functional.hartree import Hartree

def hartree_energy(density):
    ene = Hartree.compute(density, calcType=['E']).energy
    ene = density.grid.mp.asum(ene)
    return ene
