import numpy as np

from edftpy.mpi import sprint

def get_total_forces(drivers = None, gsystem = None, linearii=True, shift = True):
    forces = gsystem.get_forces(linearii = linearii)
    # sprint('Total forces0 : \n', forces)
    for i, driver in enumerate(drivers):
        if driver is None : continue
        fs = driver.get_forces()
        ind = driver.subcell.ions_index
        if driver.technique == 'OF' :
            forces[ind] += fs
        elif driver.comm.rank == 0 :
            forces[ind] += fs
        # sprint('ind\n', ind, comm = driver.comm)
        # sprint('fs\n', fs, comm = driver.comm)
    forces = gsystem.grid.mp.vsum(forces)
    #-----------------------------------------------------------------------
    if shift :
        forces_shift = np.mean(forces, axis = 0)
        sprint('Forces shift :', forces_shift)
        forces -= forces_shift
    #-----------------------------------------------------------------------
    sprint('Total forces :')
    sprint(forces)
    return forces

def get_total_stress(drivers = None, gsystem = None, **kwargs):
    stress = gsystem.get_stress()
    for i, driver in enumerate(drivers):
        if driver is None : continue
        fs = driver.get_stress()
        # if driver.comm.rank > 0 : fs = 0.0
        # sprint('fs', fs, flush=True, comm=driver.comm)
        stress += fs
    stress = gsystem.grid.mp.vsum(stress)
    for i in range(2):
        for j in range(i+1, 3):
            stress[j,i] = stress[i,j]
    sprint('Total stress :')
    sprint(stress)
    return stress
