import numpy as np
import os
from collections import OrderedDict, ChainMap
from functools import reduce
import copy
import textwrap

from dftpy.constants import LEN_CONV, ENERGY_CONV

from edftpy import io
from edftpy.config import read_conf, write_conf
from edftpy.functional import LocalPP, KEDF, Hartree, XC
from edftpy.optimizer import Optimization, MixOptimization
from edftpy.tddft import TDDFT
from edftpy.evaluator import EmbedEvaluator, EvaluatorOF, TotalEvaluator
from edftpy.density.init_density import AtomicDensity
from edftpy.subsystem.subcell import SubCell, GlobalCell
from edftpy.mixer import LinearMixer, PulayMixer
from edftpy.mpi import GraphTopo, MP, sprint
from edftpy.utils.math import get_hash, get_formal_charge
from edftpy.subsystem.decompose import decompose_sub
from edftpy.engine.driver import DriverKS, DriverEX, DriverMM, DriverOF
from edftpy.utils.common import Grid
from edftpy.utils.math import grid_map_data


def import_drivers_conf(config):
    """
    Import the engine of different drivers

    Args:
        config: contains all variables of input

    Notes:
        Must import driver firstly, before the mpi4py
    """
    calcs = []
    for key in config :
        if key.startswith('SUB'):
            calc = config[key]['calculator']
            calcs.append(calc)
    if 'pwscf' in calcs or 'qe' in calcs :
        import qepy
    if 'castep' in calcs :
        import caspytep
    return

def config_correct(config):
    """

    Correct some missing but important variables

    Args:
        config: config

    Notes:
        Only correct some missing variables
    """
    tech_keys = {'dftpy' : 'OF', 'pwscf' : 'KS', 'qe' : 'KS', 'environ'  : 'EX', 'mbx' : 'MM'}
    subkeys = [key for key in config if key.startswith('SUB')]
    for key in subkeys :
        config[key]['technique'] = tech_keys.get(config[key]['technique'], 'KS')
        if not config[key]["prefix"] :
            config[key]["prefix"] = key.lower()

        if config[key]['density']['output'] :
            if config[key]['density']['output'].startswith('.') :
                config[key]['density']['output'] = config[key]["prefix"] + config[key]['density']['output']
        if config[key]['embedpot'] :
            if config[key]['embedpot'].startswith('.') :
                config[key]['embedpot'] = config[key]["prefix"] + config[key]['embedpot']
        config[key]['hash'] = get_hash(config[key]['cell']['index'])
    #-----------------------------------------------------------------------
    key = 'GSYSTEM'
    if config[key]['density']['output'] :
        if config[key]['density']['output'].startswith('.') :
            config[key]['density']['output'] = key.lower() + config[key]['density']['output']
    #-----------------------------------------------------------------------
    return config

def config2optimizer(config, ions = None, optimizer = None, graphtopo = None, pseudo = None, cell_change = None, append = False, **kwargs):
    if isinstance(config, dict):
        pass
    elif isinstance(config, str):
        config = read_conf(config)
    ############################## Gsystem ##############################
    keysys = "GSYSTEM"
    if ions is None :
        try :
            ions = io.read(
                config["PATH"]["cell"] +os.sep+ config[keysys]["cell"]["file"],
                format=config[keysys]["cell"]["format"],
                names=config[keysys]["cell"]["elename"])
        except Exception:
            ions = io.ase_read(config["PATH"]["cell"] +os.sep+ config[keysys]["cell"]["file"])

    if optimizer is None :
        cell_change = None
    elif cell_change is None:
        if not np.allclose(optimizer.gsystem.ions.pos.cell.lattice, ions.pos.cell.lattice):
            cell_change = None # cell_change = 'cell'
        else :
            cell_change = 'position'

    if graphtopo is None :
        if optimizer is not None :
            graphtopo = optimizer.gsystem.graphtopo
        else :
            graphtopo = GraphTopo()

    gsystem = config2gsystem(config, ions = ions, optimizer = optimizer, graphtopo=graphtopo, cell_change=cell_change, **kwargs)
    grid = gsystem.grid

    #-----------------------------------------------------------------------
    adaptive = False
    reuse_drivers = {}
    if optimizer is not None :
        if config["GSYSTEM"]["decompose"]["method"] != 'manual' :
            adaptive = True
    if adaptive :
        config, reuse_drivers = config2sub_global(config, ions, optimizer=optimizer, grid=grid)
        if len(reuse_drivers) == len(optimizer.drivers) :
            # None of subsystems changed, so use normal way to update subsystems
            adaptive = False
        else :
            cell_change = None
    else :
        config = config2nsub(config, ions)
    #-----------------------------------------------------------------------
    config = config_correct(config)
    #-----------------------------------------------------------------------
    if optimizer is not None and cell_change != 'position' :
        # except PSEUDO, clean everything
        optimizer.stop_run()
        pseudo = optimizer.gsystem.total_evaluator.funcdicts['PSEUDO']
    #-----------------------------------------------------------------------
    if optimizer is not None and adaptive :
        for key, driver in reuse_drivers.items() :
            infile = config[key]["prefix"] + '_last.snpy'
            config[key]["density"]["file"] = infile
            if driver is None : continue
            if driver.comm.rank == 0 :
                io.write(infile, driver.density, driver.subcell.ions)
        graphtopo.comm.Barrier()
        graphtopo.free_comm()
    #-----------------------------------------------------------------------
    graphtopo = config2graphtopo(config, graphtopo = graphtopo)
    if graphtopo.rank == 0 :
        # import pprint
        # pprint.pprint(config)
        write_conf('edftpy_running.json', config)
        if optimizer is None :
            for key in config :
                if key.startswith('SUB') :
                    # print('Subsytem : ', key, config[key]['technique'], config[key]['cell']['index'])
                    f_str = 'Subsytem : {} {} {}'.format(key, config[key]['technique'], config[key]['cell']['index'])
                    f_str = "\n".join(textwrap.wrap(f_str, width = 80))
                    print(f_str)
    #-----------------------------------------------------------------------
    labels = set(ions.labels)
    pplist = {}
    for key in config["PP"]:
        ele = key.capitalize()
        if ele in labels :
            pplist[ele] = config["PATH"]["pp"] +os.sep+ config["PP"][key]

    if cell_change == 'position' :
        total_evaluator = gsystem.total_evaluator
    else :
        total_evaluator = None
    total_evaluator = config2total_evaluator(config, ions, grid, pplist = pplist, total_evaluator=total_evaluator, cell_change = cell_change, pseudo = pseudo)
    gsystem.total_evaluator = total_evaluator
    infile = config[keysys]["density"]["file"]
    if infile :
        gsystem.density[:] = io.read_density(infile)
    ############################## Subsytem ##############################
    subkeys = [key for key in config if key.startswith('SUB')]
    drivers = []
    for i, keysys in enumerate(subkeys):
        if cell_change == 'position' :
            driver = optimizer.drivers[i]
        else :
            driver = None
        # print('graphtopo.isub', graphtopo.isub)
        if config[keysys]["technique"] != 'OF' and graphtopo.isub != i and graphtopo.is_mpi:
            driver = None
        else :
            if config[keysys]["technique"] == 'OF' :
                mp = gsystem.grid.mp
            else :
                mp = MP(comm = graphtopo.comm_sub, decomposition = graphtopo.decomposition)
            driver = config2driver(config, keysys, ions, grid, pplist, optimizer = optimizer, cell_change = cell_change, driver = driver, mp = mp, append = append)
            #-----------------------------------------------------------------------
            #PSEUDO was evaluated on all processors, so directly remove from embedding
            # if 'PSEUDO' in driver.evaluator.funcdicts :
            #     driver.evaluator.update_functional(remove = ['PSEUDO'])
            #     gsystem.total_evaluator.update_functional(remove = ['PSEUDO'])
            #-----------------------------------------------------------------------
        drivers.append(driver)
    #-----------------------------------------------------------------------
    if len(drivers) > 0 :
        graphtopo.build_region(grid=gsystem.grid, drivers=drivers)
    #-----------------------------------------------------------------------
    for i, driver in enumerate(drivers):
        if driver is None : continue
        if (driver.technique == 'OF' and graphtopo.is_root) or (graphtopo.isub == i and graphtopo.comm_sub.rank == 0) or graphtopo.isub is None:
            outfile = driver.prefix + '.xyz'
            io.ase_write(outfile, driver.subcell.ions, format = 'extxyz', parallel = False)
    if graphtopo.is_root :
        io.ase_write('edftpy_gsystem.xyz', ions, format = 'extxyz', parallel = False)
    #-----------------------------------------------------------------------
    optimization_options = config["OPT"].copy()
    optimization_options["econv"] *= ions.nat
    task = config["JOB"]['task']
    tddft_options = config["TD"]
    opt = Optimization(drivers = drivers, options = optimization_options, gsystem = gsystem)
    #-----------------------------------------------------------------------
    if task == 'Optmix' :
        optmix = True
        task = 'Tddft'
        if optimizer is not None :
            optimizer = optimizer.optimizer_tddft
    else :
        optmix = False
    #-----------------------------------------------------------------------
    if task == 'Tddft' :
        if optimizer is not None and cell_change == 'position' :
            optimizer.gsystem = gsystem
            optimizer.drivers = drivers
            optimizer.optimizer = opt
            optimizer.options['restart'] = 'restart'
            opt = optimizer
        else :
            opt = TDDFT(drivers = drivers, options = tddft_options, gsystem = gsystem, optimizer = opt)

    if optmix :
        opt = MixOptimization(optimizer = opt)
    return opt

def config2total_embed(config, driver = None, optimizer = None, **kwargs):
    """
    Only support KS subsystems
    """
    grid_global = optimizer.gsystem.grid
    if driver is None :
        pass
    else :
        if driver.technique != 'KS' :
            raise AttributeError("Sorry config2total_embed only support KS subsystem yet.")
        if driver.comm.rank == 0 :
            grid = Grid(lattice=grid_global.lattice, nr=grid_global.nrR, full=grid_global.full, direct = True)
            pseudo = optimizer.gsystem.total_evaluator.funcdicts['PSEUDO'].restart(duplicate=True)
            total_embed = config2total_evaluator(config, driver.subcell.ions, grid, pseudo = pseudo)
            # add core density to XC
            if 'XC' in total_embed.funcdicts :
                if driver.core_density is not None :
                    core_density = grid_map_data(driver.core_density, grid = grid)
                    total_embed.funcdicts['XC'].core_density = core_density
            density = grid_map_data(driver.density, grid = grid)
            driver.density_global = density
        else :
            total_embed = None
        driver.total_embed = total_embed
    return driver

def config2gsystem(config, ions = None, optimizer = None, graphtopo = None, cell_change = None, **kwargs):
    ############################## Gsystem ##############################
    keysys = "GSYSTEM"
    if ions is None :
        try :
            ions = io.read(
                config["PATH"]["cell"] +os.sep+ config[keysys]["cell"]["file"],
                format=config[keysys]["cell"]["format"],
                names=config[keysys]["cell"]["elename"])
        except Exception:
            ions = io.ase_read(config["PATH"]["cell"] +os.sep+ config[keysys]["cell"]["file"])

    nr = config[keysys]["grid"]["nr"]
    spacing = config[keysys]["grid"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
    ecut = config[keysys]["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    full=config[keysys]["grid"]["gfull"]
    max_prime = config[keysys]["grid"]["maxprime"]
    grid_scale = config[keysys]["grid"]["scale"]
    optfft = config[keysys]["grid"]["optfft"]

    if cell_change == 'position' :
        gsystem = optimizer.gsystem
        gsystem.restart(grid=gsystem.grid, ions=ions)
        mp_global = gsystem.grid.mp
    else :
        mp_global = MP(comm = graphtopo.comm, parallel = graphtopo.is_mpi, decomposition = graphtopo.decomposition)
        gsystem = GlobalCell(ions, grid = None, ecut = ecut, nr = nr, spacing = spacing, full = full, optfft = optfft, max_prime = max_prime, scale = grid_scale, mp = mp_global, graphtopo = graphtopo)
    return gsystem

def config2graphtopo(config, graphtopo = None, scale = None):
    """
    Base on config generate new graphtopo

    Args:
        config: dict
        graphtopo: graphtopo

    Note :
        If change the size of subsystem, should free the comm_sub before this.
    """
    if graphtopo is None :
        graphtopo = GraphTopo()

    if graphtopo.comm_sub == graphtopo.comm :
        # if not initialize the comm_sub
        subkeys = [key for key in config if key.startswith('SUB')]
        nprocs = []
        #OF driver set the procs to 0, make sure it use all resources
        for key in subkeys :
            if config[key]["technique"] == 'OF' :
                n = 0
            else :
                n = config[key]["nprocs"]
            nprocs.append(n)
        graphtopo.distribute_procs(nprocs, scale = scale)
        sprint('Communicators recreated : ', graphtopo.comm.size, comm = graphtopo.comm)
    else :
        sprint('Communicators already created : ', graphtopo.comm.size, comm = graphtopo.comm)
    sprint('Number of subsystems : ', len(graphtopo.nprocs), comm = graphtopo.comm)
    f_str = np.array2string(graphtopo.nprocs, separator=' ', max_line_width=80)
    f_str += '\nUsed of processors and remainder : {} {}'.format(np.sum(graphtopo.nprocs), graphtopo.size-np.sum(graphtopo.nprocs))
    sprint('Number of processors for each subsystem : \n ', f_str, comm = graphtopo.comm)
    return graphtopo

def config2total_evaluator(config, ions, grid, pplist = None, total_evaluator= None, cell_change = None, pseudo = None):
    keysys = "GSYSTEM"
    pme = config["MATH"]["linearie"]
    xc_kwargs = config[keysys]["exc"].copy()
    ke_kwargs = config[keysys]["kedf"].copy()
    #---------------------------Functional----------------------------------
    if pseudo is not None :
        pseudo.restart(grid=grid, ions=ions, full=False)

    if cell_change == 'position' and total_evaluator is not None:
        if pseudo is None :
            pseudo = total_evaluator.funcdicts['PSEUDO']
            pseudo.restart(grid=grid, ions=ions, full=False)
        total_evaluator.funcdicts['PSEUDO'] = pseudo
    else :
        if pseudo is None :
            pseudo = LocalPP(grid = grid, ions=ions, PP_list=pplist, PME=pme)
        hartree = Hartree()
        xc = XC(**xc_kwargs)
        funcdicts = {'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
        if ke_kwargs['kedf'] is None or ke_kwargs['kedf'].lower().startswith('no'):
            pass
        else :
            ke = KEDF(**ke_kwargs)
            funcdicts['KE'] = ke
        total_evaluator = TotalEvaluator(**funcdicts)
    return total_evaluator

def config2embed_evaluator(config, keysys, ions, grid, pplist = None, cell_change = None):
    emb_ke_kwargs = config['GSYSTEM']["kedf"].copy()
    emb_xc_kwargs = config['GSYSTEM']["exc"].copy()
    pme = config["MATH"]["linearie"]

    ke_kwargs = config[keysys]["kedf"].copy()
    embed = config[keysys]["embed"]
    exttype = config[keysys]["exttype"]

    opt_options = config[keysys]["opt"].copy()
    calculator = config[keysys]["calculator"]
    #Embedding Functional---------------------------------------------------
    if exttype is not None :
        if exttype == -1 : exttype = 0
        embed = ['KE']
        if not exttype & 1 : embed.append('PSEUDO')
        if not exttype & 2 : embed.append('HARTREE')
        if not exttype & 4 : embed.append('XC')

    emb_funcdicts = {}
    if 'KE' in embed :
        if emb_ke_kwargs['kedf'] is None or emb_ke_kwargs['kedf'].lower().startswith('no'):
            pass
        else :
            ke_emb = KEDF(**emb_ke_kwargs)
            emb_funcdicts['KE'] = ke_emb
    exttype = 7
    if 'XC' in embed :
        xc_emb = XC(**emb_xc_kwargs)
        emb_funcdicts['XC'] = xc_emb
        exttype -= 4
    if 'HARTREE' in embed :
        hartree = Hartree()
        emb_funcdicts['HARTREE'] = hartree
        exttype -= 2
    if 'PSEUDO' in embed :
        pseudo = LocalPP(grid = grid, ions=ions,PP_list=pplist,PME=pme)
        emb_funcdicts['PSEUDO'] = pseudo
        exttype -= 1

    if calculator == 'dftpy' and opt_options['opt_method'] == 'full' :
        # Remove the vW part from the KE
        ke_kwargs['y'] = 0.0
        ke_kwargs['gga_remove_vw'] = True
        ke_evaluator = KEDF(**ke_kwargs)
    else :
        ke_evaluator = None

    embed_evaluator = EmbedEvaluator(ke_evaluator = ke_evaluator, **emb_funcdicts)
    return embed_evaluator, exttype

def config2evaluator_of(config, keysys, ions=None, grid=None, pplist = None, gsystem = None, cell_change = None):
    ke_kwargs = config[keysys]["kedf"].copy()
    xc_kwargs = config[keysys]["exc"].copy()
    pme = config["MATH"]["linearie"]

    embed = config[keysys]["embed"]
    exttype = config[keysys]["exttype"]

    opt_options = config[keysys]["opt"].copy()

    if exttype is not None :
        if exttype == -1 : exttype = 1
        embed = ['KE']
        if not exttype & 1 : embed.append('PSEUDO')
        if not exttype & 2 : embed.append('HARTREE')
        if not exttype & 4 : embed.append('XC')

    ke_sub_kwargs = {'name' :'vW'}
    ke_sub = KEDF(**ke_sub_kwargs)

    sub_funcdicts = {}
    if 'XC' in embed :
        xc_sub = XC(**xc_kwargs)
        sub_funcdicts['XC'] = xc_sub

    if 'HARTREE' in embed :
        hartree = Hartree()
        sub_funcdicts['HARTREE'] = hartree

    if 'PSEUDO' in embed :
        pseudo = LocalPP(grid = grid, ions=ions,PP_list=pplist,PME=pme)
        sub_funcdicts['PSEUDO'] = pseudo

    if opt_options['opt_method'] == 'full' :
        sub_funcdicts['KE'] = ke_sub
        evaluator_of = EvaluatorOF(gsystem = gsystem, **sub_funcdicts)
    else :
        # Remove the vW part from the KE
        ke_kwargs['y'] = 0.0
        ke_kwargs['gga_remove_vw'] = True
        ke_evaluator = KEDF(**ke_kwargs)
        sub_funcdicts['KE'] = ke_evaluator
        evaluator_of = EvaluatorOF(gsystem = gsystem, ke_evaluator = ke_sub, **sub_funcdicts)

    return evaluator_of

def config2driver(config, keysys, ions, grid, pplist = None, optimizer = None, cell_change = None, driver = None, subcell = None, mp = None, comm = None, append = False):
    gsystem_ecut = config['GSYSTEM']["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    pp_path = config["PATH"]["pp"]

    ecut = config[keysys]["grid"]["ecut"]
    calculator = config[keysys]["calculator"]
    mix_kwargs = config[keysys]["mix"].copy()
    basefile = config[keysys]["basefile"]
    prefix = config[keysys]["prefix"]
    kpoints = config[keysys]["kpoints"]
    exttype = config[keysys]["exttype"]
    atomicfiles = config[keysys]["density"]["atomic"].copy()
    opt_options = config[keysys]["opt"].copy()
    tddft = config['TD']
    if atomicfiles :
        for k, v in atomicfiles.copy().items():
            if not os.path.exists(v):
                atomicfiles[k] = pp_path + os.sep + v
    #-----------------------------------------------------------------------
    if ecut :
        ecut *= ENERGY_CONV["eV"]["Hartree"]
    #-----------------------------------------------------------------------
    if subcell is None :
        subcell = config2subcell(config, keysys, ions, grid, pplist = pplist, optimizer = optimizer, cell_change = cell_change, driver = driver, mp = mp, comm = comm)
    #-----------------------------------------------------------------------
    if mix_kwargs['predecut'] : mix_kwargs['predecut'] *= ENERGY_CONV["eV"]["Hartree"]
    if mix_kwargs['scheme'] == 'Pulay' :
        mixer = PulayMixer(**mix_kwargs)
    elif mix_kwargs['scheme'] == 'Linear' :
        mixer = LinearMixer(**mix_kwargs)
    elif mix_kwargs['scheme'] is None or mix_kwargs['scheme'].lower().startswith('no'):
        mixer = mix_kwargs['coef']
    else :
        raise AttributeError("!!!ERROR : NOT support ", mix_kwargs['scheme'])
    #-----------------------------------------------------------------------
    embed_evaluator, exttype = config2embed_evaluator(config, keysys, subcell.ions, subcell.grid, pplist = pplist, cell_change = cell_change)
    ncharge = config[keysys]["density"]["ncharge"]
    if config[keysys]["exttype"] and config[keysys]["exttype"] < 0 :
        exttype = config[keysys]["exttype"]
    #-----------------------------------------------------------------------
    if ncharge is None :
        mol_charges = config['MOL'].get('charge', {})
        numbers = subcell.ions.Z
        ncharge = get_formal_charge(numbers, data = mol_charges)
        if abs(ncharge) < 1E-6 : ncharge = None
    #-----------------------------------------------------------------------
    restart = False
    if driver is not None :
        task = driver.task
    else :
        task = 'scf'
        if config["JOB"]['task'] == 'Tddft' :
            if tddft['restart'] != 'initial' :
                task = 'optical'
                restart = tddft['restart'] == 'restart'

        if config["JOB"]['task'] == 'Optmix' :
            if tddft['restart'] == 'initial' :
                raise AttributeError("Sorry, 'Optmix' cannot start the TDDFT from 'initial'.")
            restart = tddft['restart'] == 'restart'
            task = config[keysys]["task"] or task
        if restart :
            restart = opt_options.get('update_sleep', 0) < 1

    margs = {
            'evaluator' : embed_evaluator,
            'prefix' : prefix,
            'subcell' : subcell,
            'cell_params' : None,
            'params' : None,
            'exttype' : exttype,
            'base_in_file' : basefile,
            'mixer' : mixer,
            'comm' : mp.comm,
            'key' : keysys,
            'ncharge' : ncharge,
            'task' : task,
            'restart' : restart,
            'append' : append,
            'options' : opt_options
            }

    if cell_change == 'cell' :
        raise AttributeError("Not support change the cell")

    if cell_change == 'position' :
        if task == 'optical' :
            driver.update_workspace(subcell, update = 1)
        else :
            driver.update_workspace(subcell)
    else :
        if calculator == 'dftpy' :
            driver = get_dftpy_driver(config, keysys, ions, grid, pplist = pplist, optimizer = optimizer, cell_change = cell_change, margs = margs)
        elif calculator == 'pwscf' or calculator == 'qe' :
            driver = get_pwscf_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)
        elif calculator == 'castep' :
            driver = get_castep_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)
        elif calculator == 'mbx' :
            driver = get_mbx_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)
        elif calculator == 'environ' :
            driver = get_environ_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)

    return driver

def get_dftpy_driver(config, keysys, ions, grid, pplist = None, optimizer = None, cell_change = None, margs = {}):
    gsystem_ecut = config['GSYSTEM']["grid"]["ecut"]
    ecut = config[keysys]["grid"]["ecut"]
    opt_options = config[keysys]["opt"].copy()
    #-----------------------------------------------------------------------
    mp = grid.mp
    subcell = margs.get('subcell')
    #-----------------------------------------------------------------------
    opt_options['econv'] *= subcell.ions.nat
    if ecut and abs(ecut - gsystem_ecut) > 1.0/ENERGY_CONV["eV"]["Hartree"]:
        if mp.comm.size > 1 :
            raise AttributeError("Different energy cutoff not supported for parallel version")
        if cell_change == 'position' :
            total_evaluator = optimizer.evaluator_of.gsystem.total_evaluator
            gsystem_driver = optimizer.evaluator_of.gsystem
        else :
            total_evaluator = None
            graphtopo = GraphTopo()
            config['GSYSTEM']["grid"]["ecut"] = ecut
            gsystem_driver = config2gsystem(config, ions = ions, optimizer = optimizer, graphtopo=graphtopo, cell_change=cell_change)
            config['GSYSTEM']["grid"]["ecut"] = gsystem_ecut
        total_evaluator = config2total_evaluator(config, ions, gsystem_driver.grid, pplist = pplist, total_evaluator=total_evaluator, cell_change = cell_change)
        gsystem_driver.total_evaluator = total_evaluator
        grid_sub = gsystem_driver.grid
        grid_sub.shift = np.zeros(3, dtype = 'int32')
    else :
        grid_sub = None
        gsystem_driver = None

    evaluator_of = config2evaluator_of(config, keysys, subcell.ions, subcell.grid, pplist = pplist, gsystem = gsystem_driver, cell_change = cell_change)

    add = {
            'options': opt_options,
            'evaluator_of': evaluator_of,
            'grid': grid_sub,
            }

    margs.update(add)
    driver = DriverOF(**margs)
    return driver

def config2subcell(config, keysys, ions, grid, pplist = None, optimizer = None, cell_change = None, driver = None, mp = None, comm = None):
    gsystem_ecut = config['GSYSTEM']["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    pp_path = config["PATH"]["pp"]

    ecut = config[keysys]["grid"]["ecut"]
    nr = config[keysys]["grid"]["nr"]
    max_prime = config[keysys]["grid"]["maxprime"]
    grid_scale = config[keysys]["grid"]["scale"]
    cellcut = config[keysys]["cell"]["cut"]
    cellsplit= config[keysys]["cell"]["split"]
    index = config[keysys]["cell"]["index"]
    use_gaussians = config[keysys]["density"]["use_gaussians"]
    gaussians_rcut = config[keysys]["density"]["gaussians_rcut"]
    gaussians_sigma = config[keysys]["density"]["gaussians_sigma"]
    gaussians_scale = config[keysys]["density"]["gaussians_scale"]
    calculator = config[keysys]["calculator"]
    technique = config[keysys]["technique"]
    optfft = config[keysys]["grid"]["optfft"]
    initial = config[keysys]["density"]["initial"]
    infile = config[keysys]["density"]["file"]
    atomicfiles = config[keysys]["density"]["atomic"].copy()
    if atomicfiles :
        for k, v in atomicfiles.copy().items():
            if not os.path.exists(v):
                atomicfiles[k] = pp_path + os.sep + v
    #-----------------------------------------------------------------------
    if ecut :
        ecut *= ENERGY_CONV["eV"]["Hartree"]
    if cellcut :
        cellcut = np.array(cellcut) * LEN_CONV["Angstrom"]["Bohr"]
    if gaussians_rcut :
        gaussians_rcut *= LEN_CONV["Angstrom"]["Bohr"]
    #-----------------------------------------------------------------------
    gaussian_options = {}
    if use_gaussians :
        for key in pplist :
            if key in gaussians_scale :
                scale = float(gaussians_scale[key])
            else :
                for i in range(ions.nat):
                    if ions.labels[i] == key :
                        break
                scale = ions.Z[i] - ions.Zval[key]
            gaussian_options[key] = {'rcut' : gaussians_rcut, 'sigma' : gaussians_sigma, 'scale' : scale}
    #-----------------------------------------------------------------------
    if calculator == 'dftpy' and ecut and abs(ecut - gsystem_ecut) > 1.0 :
        cellcut = [0.0, 0.0, 0.0]
    if cell_change == 'position' :
        grid_sub = driver.subcell.grid
    else :
        grid_sub = None

    kwargs = {
            'index' : index,
            'cellcut' : cellcut,
            'cellsplit' : cellsplit,
            'optfft' : optfft,
            'gaussian_options' : gaussian_options,
            'grid_sub' : grid_sub,
            'max_prime' : max_prime,
            'scale' : grid_scale,
            'nr' : nr,
            'mp' : mp
            }

    subcell = SubCell(ions, grid, **kwargs)

    if cell_change == 'position' :
        if subcell.density.shape == driver.density.shape :
            subcell.density[:] = driver.density
    else :
        if infile : # initial='Read'
            ext = os.path.splitext(infile)[1].lower()
            if ext != ".snpy":
                fstr = f'!WARN : snpy format for density initialization will better, but this file is "{infile}".'
                sprint(fstr, comm=subcell.grid.mp.comm, level=2)
                if subcell.grid.mp.comm.rank == 0 :
                    density = io.read_density(infile)
                else :
                    density = np.zeros(1)
                subcell.grid.scatter(density, out = subcell.density)
            else :
                subcell.density[:] = io.read_density(infile, grid=subcell.grid)
        elif initial == 'Atomic' and len(atomicfiles) > 0 :
            atomicd = AtomicDensity(files = atomicfiles)
            subcell.density[:] = atomicd.guess_rho(subcell.ions, subcell.grid)
        elif initial == 'Heg' :
            atomicd = AtomicDensity()
            subcell.density[:] = atomicd.guess_rho(subcell.ions, subcell.grid)
        elif technique == 'OF' :
            from edftpy.engine.engine_dftpy import EngineDFTpy
            subcell.density[:] = EngineDFTpy.dftpy_opt(subcell.ions, subcell.density, pplist)
        else :
            # given a negative value which means will get from driver
            subcell.density[:] = -1.0

    return subcell

def config2grid_sub(config, keysys, ions, grid, grid_sub = None, mp = None):
    ecut = config[keysys]["grid"]["ecut"]
    nr = config[keysys]["grid"]["nr"]
    max_prime = config[keysys]["grid"]["maxprime"]
    grid_scale = config[keysys]["grid"]["scale"]
    cellcut = config[keysys]["cell"]["cut"]
    cellsplit= config[keysys]["cell"]["split"]
    index = config[keysys]["cell"]["index"]
    optfft = config[keysys]["grid"]["optfft"]
    calculator = config[keysys]["calculator"]
    #-----------------------------------------------------------------------
    if cellcut :
        cellcut = np.array(cellcut) * LEN_CONV["Angstrom"]["Bohr"]
    #-----------------------------------------------------------------------
    if calculator == 'dftpy' and ecut :
        gsystem_ecut = config['GSYSTEM']["grid"]["ecut"]
        if abs(ecut - gsystem_ecut) > 1.0/ENERGY_CONV["eV"]["Hartree"]:
            cellcut = [0.0, 0.0, 0.0]

    kwargs = {
            'index' : index,
            'cellcut' : cellcut,
            'cellsplit' : cellsplit,
            'optfft' : optfft,
            'grid_sub' : grid_sub,
            'max_prime' : max_prime,
            'scale' : grid_scale,
            'nr' : nr,
            'mp' : mp
            }
    grid_sub = SubCell.gen_grid_sub(ions, grid, **kwargs)
    return grid_sub

def get_castep_driver(pplist, gsystem_ecut = None, ecut = None, kpoints = {}, margs = {}, **kwargs):
    cell_params = {'species_pot' : pplist}
    if kpoints['grid'] is not None :
        cell_params['kpoints_mp_grid'] = ' '.join(map(str, kpoints['grid']))

    params = {}
    if ecut :
        subcell = margs['subcell']
        params['cut_off_energy'] = ecut * ENERGY_CONV['Hartree']['eV']
        if abs(ecut - gsystem_ecut) < 5.0 :
            params['devel_code'] = 'STD_GRID={0} {1} {2}  FINE_GRID={0} {1} {2}'.format(*subcell.grid.nrR)

    add = {
            'cell_params': cell_params,
            'params': params,
            }
    margs.update(add)
    from edftpy.driver.ks_castep import CastepKS
    driver = CastepKS(**margs)
    return driver

def get_pwscf_driver(pplist, gsystem_ecut = None, ecut = None, kpoints = {}, margs = {}, **kwargs):
    cell_params = {'pseudopotentials' : pplist}
    if kpoints['grid'] is not None :
        cell_params['kpts'] = kpoints['grid']
    if kpoints['offset'] is not None :
        cell_params['koffset'] = kpoints['grid']

    params = {'system' : {}}
    if ecut :
        subcell = margs['subcell']
        params['system']['ecutwfc'] = ecut / 4.0 * 2.0 # 4*ecutwfc to Ry
        if abs(ecut - gsystem_ecut) < 5.0 :
            params['system']['nr1'] = subcell.grid.nrR[0]
            params['system']['nr2'] = subcell.grid.nrR[1]
            params['system']['nr3'] = subcell.grid.nrR[2]

    ncharge = margs.get('ncharge')
    if ncharge :
        params['system']['tot_charge'] = ncharge

    add = {
            'cell_params': cell_params,
            'params': params,
            }
    margs.update(add)
    # from edftpy.engine.ks_pwscf import PwscfKS
    # driver = PwscfKS(**margs)
    from edftpy.engine.engine_qe import EngineQE
    engine = EngineQE()
    driver = DriverKS(**margs, engine = engine)
    return driver

def get_environ_driver(pplist, gsystem_ecut = None, ecut = None, kpoints = {}, margs = {}, **kwargs):
    from edftpy.engine.engine_environ import EngineEnviron
    engine = EngineEnviron()
    driver = DriverEX(**margs, engine = engine)
    return driver

def get_mbx_driver(pplist, margs = {}, **kwargs):
    from edftpy.engine.engine_mbx import EngineMBX
    engine = EngineMBX()
    driver = DriverMM(**margs, engine = engine)
    return driver

def _get_gap(config, optimizer):
    for key in config :
        if key.startswith('SUB'):
            config[key]['opt']['opt_method'] = 'hamiltonian'
    config['OPT']['maxiter'] = 10
    optimizer = config2optimizer(config, optimizer.gsystem.ions, optimizer)
    optimizer.optimize()

def config2nsub(config, ions):
    subkeys = [key for key in config if key.startswith('SUB')]
    for key in subkeys :
        decompose = config[key]["decompose"]
        if decompose['method'] == 'manual' :
            pass
        elif decompose['method'] == 'distance' :
            config['N' + key] = config.pop(key)
        else :
            raise AttributeError("{} is not supported".format(decompose['method']))
    nsubkeys = [key for key in config if key.startswith('NSUB')]
    ions_inds = np.arange(ions.nat)
    for keysys in nsubkeys :
        index = config[keysys]["cell"]["index"]
        ions_sub = ions[index]

        decompose = config[keysys]["decompose"]
        indices = decompose_sub(ions_sub, decompose)

        if index is None :
            ions_inds_sub = ions_inds
        else :
            ions_inds_sub = ions_inds[index]

        for i, ind in enumerate(indices) :
            indices[i] = ions_inds_sub[ind]

        config = config_from_index(config, keysys, indices)
        # remove 'NSUB'
        del config[keysys]
    config = config2json(config, ions)
    # Only update once in the beginning.
    asub = config.get('ASUB', None)
    if not asub : config = config2asub(config)
    return config

def config_from_index(config, keysys, indices):
    # removed last step saved subs
    subs = config[keysys]['subs']
    if subs is not None :
        for key in subs : del config[key]

    prefix = config[keysys]["prefix"]
    if not prefix : prefix = keysys[1:].lower()
    subs = []
    for i, ind in enumerate(indices) :
        key = keysys[1:] + '_' + str(i)
        config[key] = copy.deepcopy(config[keysys])
        config[key]["prefix"] = prefix + '_' + str(i)
        config[key]['decompose']['method'] = 'manual'
        config[key]['cell']['index'] = ind
        config[key]["nprocs"] = max(1, config[keysys]["nprocs"] / len(indices))
        config[key]['subs'] = None
        subs.append(key)
    config[keysys]['subs'] = subs
    return config

def config2json(config, ions):
    """
    Note :
        fix some python types that not supported by json.dump
    """
    #-----------------------------------------------------------------------
    # slice and np.ndarray
    # config_json = copy.deepcopy(config)
    config_json = config
    subkeys = [key for key in config_json if key.startswith(('SUB', 'NSUB'))]
    index_w = np.arange(0, ions.nat)
    for keysys in subkeys :
        index = config_json[keysys]["cell"]["index"]
        if isinstance(index, slice):
            index = index_w[index]
        elif index is None :
            index = index_w
        if isinstance(index, np.ndarray):
            config_json[keysys]["cell"]["index"] = index.tolist()
    #-----------------------------------------------------------------------
    return config_json

def config2asub(config):
    config['ASUB'] = {}
    subkeys = [key for key in config if key.startswith('SUB')]
    for keysys in subkeys :
        #
        if not config[keysys]["prefix"] :
            config[keysys]["prefix"] = keysys.lower()
        #
        hs = get_hash(config[keysys]['cell']['index'])
        key = keysys.lower()
        config['ASUB'][key] = copy.deepcopy(config[keysys])
        config['ASUB'][key]['hash'] = hs
    return config

def config2sub_global(config, ions, optimizer = None, grid = None, regions = None):
    keysys = "GSYSTEM"
    decompose = config[keysys]["decompose"]
    if decompose['method'] == 'distance' :
        indices = decompose_sub(ions, decompose)
    else :
        raise AttributeError("{} is not supported".format(decompose['method']))

    if grid is None :
        if optimizer is None :
            raise AttributeError("Please give the grid")
        grid = optimizer.grid

    # backup last step saved subs
    subkeys = [key for key in config if key.startswith('SUB')]
    for key in subkeys :
        config['B' + key] = config.pop(key)
    subkeys_prev = [key for key in config if key.startswith('BSUB')]

    asub = config['ASUB']
    subkeys = [key for key in asub if key.startswith('sub_')]
    reuse_drivers = {}

    for i, index in enumerate(indices) :
        hs = OrderedDict()
        ind = set(index)
        nmax = 0
        na = 0
        kind = None
        this_hash = get_hash(ind)
        # check saved keys
        for keysys in subkeys_prev :
            if this_hash == config[keysys]['hash'] :
                kind = 'saved'
                this_key = keysys
                break
        # check ASUB
        if kind is None :
            for keysys in subkeys :
                base = set(asub[keysys]['cell']['index'])
                ins = base & ind
                ns = len(ins)
                if ns >0 :
                    hs[keysys] = ns
                    if ns>nmax :
                        nmax = ns
                        this_key = keysys
                    na += ns
                    # check the kind
                    if na == len(ind) :
                        if na == ns :
                            if ns < len(base):
                                kind = 'part'
                            else :
                                kind = 'one'
                        else :
                            kind = 'more'
                        break
        if kind == 'saved' :
            #the subsystem is same as last step
            #-----------------------------------------------------------------------
            key0 = this_key[1:]
            key = key0
            ik = 0
            while key in config :
                ik += 1
                key = key0 + '_' + str(ik)
            config[key] = config.pop(this_key)
            #-----------------------------------------------------------------------
            subkeys_prev.remove(this_key)
            reuse_drivers[key] = None
            if optimizer is not None :
                for driver in optimizer.drivers :
                    if driver is None : continue
                    if driver.prefix == config[key]["prefix"] :
                        reuse_drivers[key] = driver
            config[key]["prefix"] = key.lower()
        else :
            key0 = asub[this_key]["prefix"].upper()
            key = key0
            ik = 0
            while key in config :
                ik += 1
                key = key0 + '_' + str(ik)
            config[key] = copy.deepcopy(asub[this_key])
            config[key]["prefix"] = key.lower()
            if kind == 'one' :
                #just one subsystem
                pass
            elif kind == 'part' :
                #only a part of one subsystem, still use same setting as whole one
                config[key]["nprocs"] = max(1, config[key]["nprocs"]*len(index)/len(config[key]["cell"]["index"]))
                config[key]["cell"]["index"] = index
            elif kind == 'more' :
                #more than one subsystem
                nprocs = 0
                for item, nc in hs.items() :
                    nprocs += asub[item]["nprocs"]*nc/len(asub[item]["cell"]["index"])
                config[key]["nprocs"] = max(1, nprocs)
                config[key]["cell"]["index"] = index
                if regions is None :
                    regions = config2asub_region(config, ions, grid)
                lb = reduce(np.minimum, [v[0] for k, v in regions.items()])
                ub = reduce(np.maximum, [v[1] for k, v in regions.items()])
                cellsplit = np.minimum((ub - lb)/grid.nrR, 1.0)
                config[key]["cell"]["split"] = cellsplit.tolist()
            else :
                raise AttributeError("ERROR : Not support this kind : {}".format(kind))
        # set the initial density as None
        config[key]["density"]["file"] = None
        # sprint('New Subsytem : ', key, kind, index)
        f_str = 'New Subsytem : {} {} {}'.format(key, kind, index)
        f_str = "\n".join(textwrap.wrap(f_str, width = 80))
        sprint(f_str)

    # removed last step saved but unused subs
    for key in subkeys_prev :
        del config[key]
    config = config2json(config, ions)
    return (config, reuse_drivers)

def config2asub_region(config, ions, grid):
    asub = config['ASUB']
    subkeys = [key for key in asub if key.startswith('sub_')]
    regions = {}
    for keysys in subkeys :
        grid_sub = config2grid_sub(asub, keysys, ions, grid)
        lb = grid_sub.shift.copy()
        lb %= grid.nrR
        ub = lb + grid_sub.nrR
        regions[keysys] = [lb, ub]
    return regions
