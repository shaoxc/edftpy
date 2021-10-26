import numpy as np
import os
import contextlib

from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV, ZERO
from edftpy.io import write, print2file
from edftpy.properties import get_electrostatic_potential

from edftpy.api.parse_config import config2optimizer, config2total_embed
from edftpy.mpi import graphtopo, sprint

import edftpy
import dftpy

@print2file(fileobj = '/dev/null')
def import_drivers(calcs = {}):
    """
    Import the engine of different drivers

    Notes:
        Must import driver firstly, before the mpi4py
    """
    if 'GSYSTEM' in calcs :
        config = calcs
        calcs = []
        for key in config :
            if key.startswith('SUB'):
                calc = config[key]['calculator']
                calcs.append(calc)
    fs = "{:>20s} Version : {}\n"
    info = fs.format('eDFTpy', edftpy.__version__)
    info += fs.format('DFTpy', dftpy.__version__)
    #
    try:
        from edftpy.engine import engine_qe
        if 'pwscf' in calcs or 'qe' in calcs :
            info += fs.format('QEpy', engine_qe.__version__)
    except Exception as e:
        if 'pwscf' in calcs or 'qe' in calcs :
            raise AttributeError(e+"\nPlease install 'QEpy' firstly.")
    try:
        from edftpy.engine import engine_castep
        if 'castep' in calcs :
            info += fs.format('Caspytep', engine_castep.__version__)
    except Exception as e:
        if 'castep' in calcs :
            raise AttributeError(e+"\nPlease install 'caspytep' firstly.")
    try:
        # Because Environ use same parallel technique as QE, need import after QEpy.
        from edftpy.engine import engine_environ
        if 'environ' in calcs :
            info += fs.format('Environ', engine_environ.__version__)
    except Exception as e:
        if 'environ' in calcs :
            raise AttributeError(e+"\nPlease install 'environ' firstly.")
    return info

def conf2init(conf, parallel = False, **kwargs):
    info = import_drivers(conf)
    graphtopo = init_graphtopo(parallel, info = info, **kwargs)
    return graphtopo

def init_graphtopo(parallel = False, info = None, **kwargs):
    header = '*'*80 + '\n'
    if parallel :
        try :
            from mpi4py import MPI
            from mpi4py_fft import PFFT
            graphtopo.comm = MPI.COMM_WORLD
            header += 'Parallel version (MPI) on {0:>8d} processors\n'.format(graphtopo.comm.size)
        except Exception as e:
            raise e
    else :
        header += 'Serial version on {0:>8d} processor\n'.format(1)
    if graphtopo.rank == 0 :
        #-----------------------------------------------------------------------
        # remove the stopfile
        if os.path.isfile('edftpy_stopfile'): os.remove('edftpy_stopfile')
        #-----------------------------------------------------------------------
    if info is None :
        fs = "{:>20s} Version : {}\n"
        header += fs.format('eDFTpy', edftpy.__version__)
        header += fs.format('DFTpy', dftpy.__version__)
    else :
        header += info
    header += '*'*80
    sprint(header)
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

def optimize_embed(config, optimizer, lprint = False, **kwargs):
    if not lprint :
        subkeys = [key for key in config if key.startswith('SUB')]
        for keysys in subkeys:
            if config[keysys].get("embedpot", None):
                lprint = True
                break
    if lprint :
        optimizer.set_kedf_params()
        # remove = ['PSEUDO', 'HARTREE', 'KE']
        remove = []
        removed = optimizer.gsystem.total_evaluator.update_functional(remove = remove)
        optimizer.set_global_potential()
        optimizer.gsystem.total_evaluator.update_functional(add = removed)
        for i, driver in enumerate(optimizer.drivers):
            if driver is None : continue
            outfile = config[driver.key]["embedpot"]
            if outfile :
                driver = config2total_embed(config, driver = driver, optimizer = optimizer)
                if driver.technique == 'OF' or driver.comm.rank == 0 or graphtopo.isub is None:
                    removed_sub = driver.total_embed.update_functional(remove = remove)
                    potential = driver.total_embed(driver.density_global, calcType = ['V']).potential
                    index = optimizer.gsystem.graphtopo.graph.get_sub_index(i, in_global = True)
                    potential = driver.evaluator.global_potential - potential[index]
                    write(outfile, potential, ions = driver.subcell.ions, data_type = 'potential')
                    driver.total_embed.update_functional(add = removed_sub)

    return

def conf2output(config, optimizer):
    optimize_embed(config, optimizer)
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

    if config["OUTPUT"]["sub_temp"] :
        save = ['D', 'W']
    else :
        save = ['D']
    optimizer.stop_run(save = save)
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
