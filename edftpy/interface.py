import numpy as np

from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV, ZERO
from edftpy.io import write
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
    #-----------------------------------------------------------------------
    kefunc = opt.gsystem.total_evaluator.funcdicts.get('KE', None)
    if kefunc is not None and kefunc.name.startswith('MIX_'):
        opt.set_kedf_params(level = -1)
        for i, driver in enumerate(opt.drivers):
            if driver is None : continue
            driver.update_workspace(first = True)
        opt.optimize()
    #-----------------------------------------------------------------------
    energy = opt.energy
    sprint('Final energy (a.u.)', energy)
    sprint('Final energy (eV)', energy * ENERGY_CONV['Hartree']['eV'])
    sprint('Final energy (eV/atom)', energy * ENERGY_CONV['Hartree']['eV']/opt.gsystem.ions.nat)
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

    if "Force" in config["JOB"]["calctype"]:
        sprint("Calculate Force...")
        forces = optimizer.get_forces()
        ############################## Output Force ##############################
        if optimizer.gsystem.grid.mp.rank == 0 :
            sprint("-" * 80)
            fabs = np.abs(forces)
            fmax, fmin, fave = fabs.max(axis = 0), fabs.min(axis = 0), fabs.mean(axis = 0)
            fmsd = (fabs * fabs).mean(axis = 0)
            fstr_f = " " * 4 + "{0:>12s} : {1:< 22.10f} {2:< 22.10f} {3:< 22.10f}"
            sprint(fstr_f.format("Max force (a.u.)", *fmax))
            sprint(fstr_f.format("Min force (a.u.)", *fmin))
            sprint(fstr_f.format("Ave force (a.u.)", *fave))
            sprint(fstr_f.format("MSD force (a.u.)", *fmsd))
            sprint("-" * 80)
        optimizer.gsystem.grid.mp.comm.Barrier()
    return

def __optimize_density_conf_test(config, **kwargs):
    opt = config2optimizer(config, **kwargs)
    opt.optimize()
    #-----------------------------------------------------------------------
    kefunc = opt.gsystem.total_evaluator.funcdicts.get('KE', None)
    if kefunc is not None and kefunc.name.startswith('MIX_'):
        opt.set_kedf_params(level = -1)
        for i, driver in enumerate(opt.drivers):
            if driver is None : continue
            driver.stop_run()
        opt2 = opt
        opt = config2optimizer(config, opt.gsystem.ions, graphtopo = opt.gsystem.graphtopo)
        #-----------------------------------------------------------------------
        for i, driver in enumerate(opt.drivers):
            if driver is None : continue
            if 'KE' in driver.evaluator.funcdicts :
                driver.evaluator.funcdicts['KE'].rhomax = opt2.drivers[i].evaluator.funcdicts['KE'].rhomax
            opt.gsystem.total_evaluator.funcdicts['KE'].rhomax = opt2.gsystem.total_evaluator.funcdicts['KE'].rhomax
        opt.optimize()
    #-----------------------------------------------------------------------
    energy = opt.energy
    sprint('Final energy (a.u.)', energy)
    sprint('Final energy (eV)', energy * ENERGY_CONV['Hartree']['eV'])
    sprint('Final energy (eV/atom)', energy * ENERGY_CONV['Hartree']['eV']/opt.gsystem.ions.nat)
    return opt
