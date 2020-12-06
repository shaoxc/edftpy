import numpy as np
from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.formats import io

from edftpy.utils.common import Grid
from edftpy.subsystem.subcell import GlobalCell
from edftpy.api.parse_config import config2optimizer
from edftpy.mpi import sprint

def optimize_density_conf(config, **kwargs):
    opt = config2optimizer(config, **kwargs)
    opt.optimize()
    energy = opt.energy
    sprint('Final energy (a.u.)', energy)
    sprint('Final energy (eV)', energy * ENERGY_CONV['Hartree']['eV'])
    # for i, driver in enumerate(opt.drivers):
        # io.write('final_sub_' + str(i) + '.xsf', driver.density, driver.subcell.ions)
    # io.write('final.xsf', opt.density, opt.gsystem.ions)
    return opt

def get_forces(drivers = None, gsystem = None, linearii=True):
    forces = gsystem.get_forces(linearii = linearii)
    sprint('Total forces0 : \n', forces)
    for i, driver in enumerate(drivers):
        fs = driver.get_forces()
        ind = driver.subcell.ions_index
        sprint('ind', ind)
        sprint('fs', fs)
        forces[ind] += fs
    sprint('Total forces : \n', forces)
    return forces

def get_stress():
    pass

def get_total_density(gsystem, drivers = None, scale = 1):
    results = []
    if scale > 1 :
        """
        Only support for serial.
        """
        grid_global = Grid(gsystem.grid.lattice, gsystem.grid.nr * scale, direct = True)
        gsystem_fine = GlobalCell(gsystem.ions, grid = grid_global)
        for i, driver in enumerate(drivers):
            grid_sub = Grid(driver.grid.lattice, driver.grid.nr * scale, direct = True)
            grid_sub.shift = driver.grid.shift * scale
            rho = driver._format_density_invert(charge = driver.charge, grid = grid_sub)
            results.append(rho)
            if i == 0 :
                restart = True
            else :
                restart = False
            gsystem_fine.update_density(rho, restart = restart)
        results.insert(0, gsystem_fine.density)
    else :
        for i, driver in enumerate(drivers):
            rho = driver.density
            results.append(rho)
            if i == 0 :
                restart = True
            else :
                restart = False
            gsystem.update_density(rho, isub = i, restart = restart)
        results.insert(0, gsystem.density)
    return results
