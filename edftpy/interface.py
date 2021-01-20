import numpy as np
from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.formats.io import write
from edftpy.properties import get_electrostatic_potential

from edftpy.utils.common import Grid
from edftpy.subsystem.subcell import GlobalCell
from edftpy.api.parse_config import config2optimizer, import_drivers
from edftpy.mpi import graphtopo, sprint, pmi

def conf2init(conf, parallel = False, *args, **kwargs):
    import_drivers(conf)
    if parallel :
        try :
            from mpi4py import MPI
            from mpi4py_fft import PFFT
            graphtopo.comm = MPI.COMM_WORLD
            info = 'Parallel version (MPI) on {0:>8d} processors'.format(graphtopo.comm.size)
        except Exception as e:
            raise e
    else :
        info = 'Serial version on {0:>8d} processor'.format(1)
    sprint(info)
    return graphtopo

def optimize_density_conf(config, **kwargs):
    opt = config2optimizer(config, **kwargs)
    opt.optimize()
    energy = opt.energy
    sprint('Final energy (a.u.)', energy)
    sprint('Final energy (eV)', energy * ENERGY_CONV['Hartree']['eV'])
    return opt

def get_forces(drivers = None, gsystem = None, linearii=True):
    forces = gsystem.get_forces(linearii = linearii)
    sprint('Total forces0 : \n', forces)
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

def conf2output(config, optimizer):
    if config["GSYSTEM"]["density"]['output']:
        sprint("Write Density...")
        outfile = config["GSYSTEM"]["density"]['output']
        write(outfile, optimizer.density, ions = optimizer.gsystem.ions)
    if config["OUTPUT"]["electrostatic_potential"]:
        sprint("Write electrostatic potential...")
        outfile = config["OUTPUT"]["electrostatic_potential"]
        v = get_electrostatic_potential(optimizer.gsystem)
        write(outfile, v, ions = optimizer.gsystem.ions)

    for i, driver in enumerate(optimizer.drivers):
        if driver is None : continue
        outfile = config[driver.key]["density"]['output']
        if outfile :
            if driver.technique == 'OF' or driver.comm.rank == 0 or graphtopo.isub is None:
                write(outfile, driver.density, ions = driver.subcell.ions)
