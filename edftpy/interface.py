import numpy as np
from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.formats.io import write
from edftpy.properties import get_electrostatic_potential

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
