import numpy as np
from dftpy.constants import FORCE_CONV, STRESS_CONV

from edftpy.mpi import sprint

def get_total_forces(drivers = None, gsystem = None, linearii=True):
    forces = gsystem.get_forces(linearii = linearii)
    # sprint('Total forces0 : \n', forces)
    for i, driver in enumerate(drivers):
        if driver is None : continue
        fs = driver.get_forces()
        if driver.comm.rank == 0 :
            ind = driver.subcell.ions_index
            # sprint('ind', ind)
            # sprint('fs', fs)
            forces[ind] += fs
    forces = gsystem.grid.mp.vsum(forces)
    sprint('Total forces : \n', forces)
    return forces

def get_total_stress(drivers = None, gsystem = None, linearii=True, **kwargs):
    pass
