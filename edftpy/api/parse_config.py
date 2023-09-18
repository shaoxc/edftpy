import numpy as np
import os
from collections import OrderedDict
from functools import reduce
import copy
import textwrap

from dftpy.constants import LEN_CONV, ENERGY_CONV

from edftpy import io
from edftpy.config import read_conf, write_conf
from edftpy.functional import LocalPP, KEDF, Hartree, XC, Ewald
from edftpy.optimizer import Optimization, MixOptimization
from edftpy.tddft import TDDFT
from edftpy.evaluator import EmbedEvaluator, EvaluatorOF, TotalEvaluator
from edftpy.density import file2density, DensityGenerator
from edftpy.subsystem.subcell import SubCell, GlobalCell
from edftpy.mixer import Mixer
from edftpy.mpi import GraphTopo, MP, sprint
from edftpy.utils.math import get_hash, get_formal_charge
from edftpy.subsystem.decompose import decompose_sub
from edftpy.engine.driver import DriverKS, DriverEX, DriverMM, DriverOF
from edftpy.utils.common import Grid, Ions
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
    if 'pwscf' in calcs or 'qe' in calcs or 'qepy' in calcs :
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
    tech_keys = {
            'dftpy' : 'OF',
            'pwscf' : 'KS',
            'qe' : 'KS',
            'qepy' : 'KS',
            'environ' : 'EX',
            'mbx' : 'MM'
            }
    subkeys = [key for key in config if key.startswith('SUB')]
    for key in subkeys :
        if config[key]['calculator'] in ['pwscf', 'qepy'] : config[key]['calculator'] = 'qe'
        config[key]['technique'] = tech_keys.get(config[key]['calculator'], 'KS')
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
    ions = config2ions(config, ions = ions)

    if optimizer is None :
        cell_change = None
    elif cell_change is None:
        if not np.allclose(optimizer.gsystem.ions.cell, ions.cell):
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
    append = append or config["OUTPUT"]["append"]
    #-----------------------------------------------------------------------
    if optimizer is not None and cell_change != 'position' :
        # except PSEUDO, clean everything
        optimizer.stop_run()
        pseudo = optimizer.gsystem.total_evaluator.funcdicts['PSEUDO']
    #-----------------------------------------------------------------------
    if optimizer is not None and adaptive :
        graphtopo.free_comm()
    #-----------------------------------------------------------------------
    graphtopo = config2graphtopo(config, graphtopo = graphtopo)
    if graphtopo.rank == 0 :
        config = ions2config(config, ions)
        write_conf('edftpy_running.json', config)
        io.write('edftpy_gsystem.xyz', ions)
        if optimizer is None :
            for key in config :
                if key.startswith('SUB') :
                    f_str = 'Subsystem : {} {} {}'.format(key, config[key]['technique'], config[key]['cell']['index'])
                    f_str = "\n".join(textwrap.wrap(f_str, width = 80))
                    sprint(f_str, lprint = True)
    #-----------------------------------------------------------------------
    labels = set(ions.symbols)
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
    ############################## Subsystem ##############################
    subkeys = [key for key in config if key.startswith('SUB')]
    drivers = []
    nr_all = np.zeros((len(subkeys), 3), dtype = np.int64)
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
            driver = config2driver(config, keysys, ions, grid, pplist, total_evaluator = total_evaluator, optimizer = optimizer, cell_change = cell_change, driver = driver, mp = mp, append = append, nspin = gsystem.density.rank)
            if mp.rank == 0 and driver.grid_driver is not None :
                nr_all[i] = driver.grid_driver.nrR
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
        nr_all = grid.mp.vsum(nr_all)
        if graphtopo.is_root :
            config_new = copy.deepcopy(config)
            config_new["GSYSTEM"]["grid"]["nr"] = gsystem.grid.nrR.tolist()
            fstr = ''
            for i, keysys in enumerate(subkeys):
                nr_shift = graphtopo.graph.sub_shift[i]
                nrs = graphtopo.graph.sub_shape[i]
                if nr_all[i][0] == 0 :
                    nrs2 = graphtopo.graph.sub_shape[i]
                else :
                    nrs2 = nr_all[i]
                config_new[keysys]["grid"]["nr"] = graphtopo.graph.sub_shape[i].tolist()
                config_new[keysys]["grid"]["shift"] = graphtopo.graph.sub_shift[i].tolist()
                fstr += f'Grid : {keysys} {nr_shift} {nrs} {nrs2}\n'
            sprint(fstr, lprint = True)
            write_conf('edftpy_running.json', config_new)
    #-----------------------------------------------------------------------
    for i, driver in enumerate(drivers):
        if driver is None : continue
        if (driver.technique == 'OF' and graphtopo.is_root) or (graphtopo.isub == i and graphtopo.comm_sub.rank == 0) or graphtopo.isub is None:
            outfile = driver.prefix + '.xyz'
            io.write(outfile, driver.subcell.ions)
    #-----------------------------------------------------------------------
    optimization_options = config["OPT"].copy()
    optimization_options["econv"] *= ions.nat
    task = config["JOB"]['task']
    tddft_options = config["TD"]
    mix_kwargs = config["GSYSTEM"]["mix"].copy()
    if mix_kwargs['predecut'] : mix_kwargs['predecut'] *= ENERGY_CONV["eV"]["Hartree"]
    mixer = Mixer(**mix_kwargs)
    opt = Optimization(drivers = drivers, options = optimization_options, gsystem = gsystem, mixer = mixer)
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

    if opt.sdft == 'qmmm' :
        indices = np.arange(gsystem.ions.nat)
        subkeys = [key for key in config if key.startswith('SUB')]
        index_mm = []
        for key in subkeys :
            if config[key]['technique'] == 'MM' :
                ind = config[key]['cell']['index']
                index_mm.extend(ind)
        #MM part
        ions_mm = ions[index_mm]
        gsystem_mm = config2gsystem(config, ions = ions_mm, graphtopo=graphtopo, grid = gsystem.grid, index = index_mm, **kwargs)
        total_evaluator_mm = config2total_evaluator(config, ions_mm, grid, pplist = pplist)
        gsystem_mm.total_evaluator = total_evaluator_mm
        #QM part
        index_qm = np.delete(indices, index_mm)
        ions_qm = ions[index_qm]
        gsystem_qm = config2gsystem(config, ions = ions_qm, graphtopo=graphtopo, grid = gsystem.grid, index = index_qm, **kwargs)
        total_evaluator_qm = config2total_evaluator(config, ions_qm, grid, pplist = pplist)
        gsystem_qm.total_evaluator = total_evaluator_qm
        #Swap QMMM and QM
        opt.gsystem_qmmm = opt.gsystem
        opt.gsystem = gsystem_qm
        opt.gsystem_mm = gsystem_mm
        #-----------------------------------------------------------------------
        #! Only for one MM subsystem
        ions_qmmm = ions.copy()
        ions_mm = ions[index_mm]
        rank_d = 0
        pos_m, inds_m, inds_o = None, None, None
        for driver in opt.drivers:
            if driver is not None and driver.technique== 'MM' :
                pos_m, inds_m, inds_o = driver.engine.get_m_sites()
                if driver.comm.rank == 0 :
                    rank_d = opt.gsystem.graphtopo.comm.rank
        rank_d = opt.gsystem.graphtopo.comm.allreduce(rank_d)
        pos_m  = opt.gsystem.graphtopo.comm.bcast(pos_m, root = rank_d)
        inds_m = opt.gsystem.graphtopo.comm.bcast(inds_m, root = rank_d)
        inds_o = opt.gsystem.graphtopo.comm.bcast(inds_o, root = rank_d)
        # print('rank', opt.gsystem.graphtopo.comm.rank, rank_d, inds_m, pos_m)
        if len(pos_m) > 0 :
            ions_mm.pos[inds_o] = pos_m
            index_mm = opt.gsystem_mm.ions_index
            ions_qmmm.pos[index_mm] = ions_mm.pos

        def update_evaluator_ions(evaluator, ions, grid = None, linearii = True):
            pseudo = evaluator.funcdicts['PSEUDO']
            if grid is None : grid = pseudo.grid
            ewald = Ewald(ions=ions, grid = grid, PME=linearii)
            evaluator.funcdicts['EWALD'] = ewald
            pseudo.restart(grid=grid, ions=ions, full=False)
            return evaluator

        linearii = config["MATH"]["linearii"]
        opt.gsystem_qmmm.total_evaluator = update_evaluator_ions(opt.gsystem_qmmm.total_evaluator, ions_qmmm, linearii = linearii)
        opt.gsystem_mm.total_evaluator = update_evaluator_ions(opt.gsystem_mm.total_evaluator, ions_mm, linearii = linearii)
        #-----------------------------------------------------------------------
    # for driver in opt.drivers:
        # if driver is not None and driver.technique== 'MM' :
            #-----------------------------------------------------------------------test
            # from edftpy.utils.common import Atoms, Field
            # charges, positions_c = driver.engine.get_charges()
            # charges = driver.engine.points_zval - charges
            # # pot = driver.engine.get_potential(grid = driver.grid_driver)
            # # pot = Field(grid = driver.grid_driver, data = pot)
            # # pot.write('0_mm_pot.xsf', ions = driver.subcell.ions)
            # # atomicd = DensityGenerator(pseudo = opt.gsystem_mm.total_evaluator.pseudo, comm = driver.comm)
            # atomicd = DensityGenerator(pseudo = opt.gsystem.total_evaluator.pseudo, comm = driver.comm)
            # atomicd._arho['O'] *= charges[3] / atomicd._arho['O'][0]
            # atomicd._arho['H'] *= charges[1] / atomicd._arho['H'][0]
            # ions = Atoms(['H', 'H', 'O'], zvals =driver.subcell.ions.Zval, pos=positions_c[1:], cell = driver.subcell.grid.lattice, basis = 'Cartesian')
            # rho = atomicd.guess_rho(ions, driver.subcell.grid)

            # # rho = atomicd.guess_rho(driver.subcell.ions, driver.subcell.grid)
            # rho.write('0_atomic_mm.xsf', ions = driver.subcell.ions)
            #-----------------------------------------------------------------------test
    return opt

def config2ions(config, ions = None, keysys = 'GSYSTEM', **kwargs):
    if ions is not None : return ions
    if config[keysys]["cell"]["file"] :
        filename = config["PATH"]["cell"] +os.sep+ config[keysys]["cell"]["file"]
        try :
            ions = io.read(filename,
                format=config[keysys]["cell"]["format"],
                names=config[keysys]["cell"]["elename"])
        except Exception:
            ions = io.ase_read(filename,
                    format=config[keysys]["cell"]["format"])
    else :
        lattice = config[keysys]['cell']['lattice']
        symbols = config[keysys]['cell']['symbols']
        positions = config[keysys]['cell']['positions']
        scaled_positions = config[keysys]['cell']['scaled_positions']
        numbers = config[keysys]['cell']['numbers']
        if len(lattice) == 9 :
            lattice = np.asarray(lattice).reshape((3, 3))
        if positions is not None :
            positions = np.asarray(positions).reshape((-1, 3))
        if scaled_positions is not None :
            scaled_positions = np.asarray(scaled_positions).reshape((-1, 3))
        ions = Ions(symbols = symbols, positions = positions, cell = lattice,
                numbers = numbers, scaled_positions = scaled_positions, units = 'ase')
    return ions

def ions2config(config, ions, keysys = 'GSYSTEM', **kwargs):
    lattice = ions.cell.ravel()
    config[keysys]['cell']['lattice'] = (lattice*LEN_CONV["Bohr"]["Angstrom"]).tolist()
    config[keysys]['cell']['symbols'] = ions.get_chemical_symbols()
    config[keysys]['cell']['positions'] = (ions.positions.ravel()*LEN_CONV["Bohr"]["Angstrom"]).tolist()
    return config

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

def config2gsystem(config, ions = None, optimizer = None, graphtopo = None, cell_change = None, grid = None, index = None, **kwargs):
    ############################## Gsystem ##############################
    keysys = "GSYSTEM"

    ions = config2ions(config, ions = ions)

    nr = config[keysys]["grid"]["nr"]
    spacing = config[keysys]["grid"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
    ecut = config[keysys]["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    full=config[keysys]["grid"]["gfull"]
    max_prime = config[keysys]["grid"]["maxprime"]
    grid_scale = config[keysys]["grid"]["scale"]
    optfft = config[keysys]["grid"]["optfft"]
    density_file = config[keysys]["density"]["file"]
    nspin = config[keysys]["density"]["nspin"]

    if cell_change == 'position' :
        gsystem = optimizer.gsystem
        gsystem.restart(grid=gsystem.grid, ions=ions)
        mp_global = gsystem.grid.mp
    else :
        mp_global = MP(comm = graphtopo.comm, parallel = graphtopo.is_mpi, decomposition = graphtopo.decomposition)
        gsystem = GlobalCell(ions, grid = grid, ecut = ecut, nr = nr, spacing = spacing, full = full, optfft = optfft, max_prime = max_prime, scale = grid_scale, mp = mp_global, graphtopo = graphtopo, nspin = nspin, index = index)

    if density_file : file2density(density_file, gsystem.density)
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
        if not nprocs : nprocs = [0]
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
    linearii = config["MATH"]["linearii"]
    xc_kwargs = config[keysys]["exc"].copy()
    ke_kwargs = config[keysys]["kedf"].copy()
    environ_kwargs = config[keysys].get('environ', {})
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
        # xc = XC(pseudo = pseudo, **xc_kwargs)
        funcdicts = {'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
        if ke_kwargs['kedf'] is None or ke_kwargs['kedf'].lower().startswith('no'):
            pass
        else :
            ke = KEDF(**ke_kwargs)
            funcdicts['KE'] = ke
        total_evaluator = TotalEvaluator(**funcdicts)
    # Only depend on atoms---------------------------------------------------
    ewald = Ewald(ions=ions, grid = grid, PME=linearii)
    total_evaluator.funcdicts['EWALD'] = ewald
    if xc_kwargs.get('dftd4', None):
        from edftpy.api.dftd4 import VDWDFTD4
        vdw = VDWDFTD4(ions = ions, mp = grid.mp, **xc_kwargs)
        total_evaluator.funcdicts['VDW'] = vdw
    #-----------------------------------------------------------------------
    if environ_kwargs.get('file', None):
        from edftpy.functional import Environ
        environ = Environ(grid=grid, ions=ions, **environ_kwargs)
        total_evaluator.funcdicts['ENVIRON'] = environ
    #-----------------------------------------------------------------------
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

def config2driver(config, keysys, ions, grid, pplist = None, total_evaluator = None, optimizer = None, cell_change = None, driver = None, subcell = None, mp = None, comm = None, append = False, nspin = 1):
    gsystem_ecut = config['GSYSTEM']["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    ecut = config[keysys]["grid"]["ecut"]
    calculator = config[keysys]["calculator"]
    mix_kwargs = config[keysys]["mix"].copy()
    basefile = config[keysys]["basefile"]
    prefix = config[keysys]["prefix"]
    kpoints = config[keysys]["kpoints"]
    exttype = config[keysys]["exttype"]
    xc_options = config[keysys]["exc"].copy()
    opt_options = config[keysys]["opt"].copy()
    density_initial = config[keysys]["density"]["initial"]
    density_file = config[keysys]["density"]["file"]
    tddft = config['TD']
    #-----------------------------------------------------------------------
    if ecut :
        ecut *= ENERGY_CONV["eV"]["Hartree"]
    #-----------------------------------------------------------------------
    if subcell is None :
        subcell = config2subcell(config, keysys, ions, grid, pplist = pplist, total_evaluator = total_evaluator, optimizer = optimizer, cell_change = cell_change, driver = driver, mp = mp, comm = comm, nspin = nspin)
    #-----------------------------------------------------------------------
    if mix_kwargs['predecut'] : mix_kwargs['predecut'] *= ENERGY_CONV["eV"]["Hartree"]
    mixer = Mixer(**mix_kwargs)
    if mixer is None : mixer = mix_kwargs.get('coef', 0.7)
    #-----------------------------------------------------------------------
    embed_evaluator, exttype = config2embed_evaluator(config, keysys, subcell.ions, subcell.grid, pplist = pplist, cell_change = cell_change)
    if config[keysys]["exttype"] and config[keysys]["exttype"] < 0 :
        exttype = config[keysys]["exttype"]
    #-----------------------------------------------------------------------
    ncharge = config[keysys]["density"]["ncharge"]
    if ncharge is None :
        mol_charges = config['MOL'].get('charge', {})
        numbers = subcell.ions.numbers
        ncharge = get_formal_charge(numbers, data = mol_charges)
        if abs(ncharge) < 1E-6 : ncharge = None
    #-----------------------------------------------------------------------
    magmom = config[keysys]["density"]["magmom"]
    if magmom is None :
        mol_magmom= config['MOL'].get('magmom', {})
        numbers = subcell.ions.numbers
        magmom = get_formal_charge(numbers, data = mol_magmom)
        if abs(magmom) < 1E-6 : magmom = None
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
    #-----------------------------------------------------------------------
    if density_file :
        density_initial = 'file'
    #-----------------------------------------------------------------------

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
            'magmom' : magmom,
            'task' : task,
            'restart' : restart,
            'append' : append,
            'options' : opt_options,
            'xc' : xc_options,
            'density_initial' : density_initial
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
        elif calculator == 'qe' :
            driver = get_pwscf_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)
        elif calculator == 'castep' :
            driver = get_castep_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)
        elif calculator == 'mbx' :
            driver = get_mbx_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)
        elif calculator == 'environ' :
            driver = get_environ_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)
        else :
            raise AttributeError(f"Not supported engine : {calculator}")
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

def config2subcell(config, keysys, ions, grid, pplist = None, total_evaluator = None, optimizer = None, cell_change = None, driver = None, mp = None, comm = None, nspin = 1):
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
    density_file = config[keysys]["density"]["file"]
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
                    if ions.symbols[i] == key :
                        break
                scale = ions.numbers[i] - ions.zval[key]
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
            'mp' : mp,
            'nspin' : nspin,
            }

    subcell = SubCell(ions, grid, **kwargs)

    if total_evaluator is not None :
        pseudo = total_evaluator.funcdicts['PSEUDO']
    else :
        pseudo = None

    if cell_change == 'position' :
        if subcell.density.shape == driver.density.shape :
            subcell.density[:] = driver.density
    else :
        if density_file : # initial='read'
            file2density(density_file, subcell.density)
        elif initial == 'atomic' :
            atomicd = DensityGenerator(files = atomicfiles, pseudo = pseudo, comm = subcell.grid.mp.comm)
            atomicd.guess_rho(subcell.ions, subcell.grid, rho = subcell.density)
        elif initial == 'heg' :
            atomicd = DensityGenerator()
            atomicd.guess_rho(subcell.ions, subcell.grid, rho = subcell.density)
        elif technique == 'OF' : # ofdft
            from edftpy.engine.engine_dftpy import EngineDFTpy
            subcell.density[:] = EngineDFTpy.dftpy_opt(subcell.ions, subcell.density, pplist, pseudo = pseudo)
        else :
            pass
    #-----------------------------------------------------------------------
    if calculator not in ['qe'] :
        atomicd = DensityGenerator(pseudo = pseudo, comm = subcell.grid.mp.comm, is_core = True)
        atomicd.guess_rho(subcell.ions, subcell.grid, rho=subcell.core_density)
    #-----------------------------------------------------------------------

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
    subcell = margs['subcell']
    if ecut :
        params['system']['ecutwfc'] = ecut / 4.0 * 2.0 # 4*ecutwfc to Ry
        if abs(ecut - gsystem_ecut) < 5.0 :
            params['system']['nr1'] = subcell.grid.nrR[0]
            params['system']['nr2'] = subcell.grid.nrR[1]
            params['system']['nr3'] = subcell.grid.nrR[2]
    params['system']['nspin'] = subcell.density.rank

    ncharge = margs.get('ncharge')
    magmom = margs.get('magmom')
    if ncharge :
        params['system']['tot_charge'] = ncharge
    if magmom:
        params['system']['tot_magnetization'] = magmom

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
    engine = EngineMBX(xc = margs.get('xc'))
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
        indices = decompose_sub(ions_sub, **decompose)

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

    # allocate resources according to the the number of atoms
    nprocs = []
    natoms = sum([len(ind) for ind in indices])
    for ind in indices :
        n = config[keysys]["nprocs"]*len(ind)/natoms
        nprocs.append(n)
    nprocs = np.asarray(nprocs)
    if nprocs.min() < 1 : nprocs /= nprocs.min()

    prefix = config[keysys]["prefix"]
    if not prefix : prefix = keysys[1:].lower()
    subs = []
    for i, ind in enumerate(indices) :
        key = keysys[1:] + '_' + str(i)
        config[key] = copy.deepcopy(config[keysys])
        config[key]["prefix"] = prefix + '_' + str(i)
        config[key]['decompose']['method'] = 'manual'
        config[key]['cell']['index'] = ind
        # config[key]["nprocs"] = max(1, config[keysys]["nprocs"]*len(ind)/natoms)
        config[key]["nprocs"] = nprocs[i]
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

def config2sub_check_update(config, ions, rtol = 0.1):
    if rtol < 1E-6 : return True
    keysys = "GSYSTEM"
    decompose = copy.deepcopy(config[keysys]["decompose"])
    if decompose['method'] != 'distance' :
        raise AttributeError("{} is not supported".format(decompose['method']))
    rcut = decompose.get('rcut', 3.0)
    radius = decompose.get('radius', {})
    if len(radius) == 0 :
        decompose['rcut'] = rcut + rtol
    else :
        for k in radius :
            decompose['radius'][k] += 0.5*rtol
    indices = decompose_sub(ions, **decompose)
    subkeys = [key for key in config if key.startswith('SUB')]
    #
    if len(subkeys) < len(indices) :
        sprint('Adaptive-Update : Have more subsystems')
        return True
    #
    old = [np.asarray(config[x]['cell']['index']) for x in subkeys]
    new = [np.asarray(x) for x in indices]
    nsub_new_1 = len(new)

    def issubset(a, b):
        for idx0 in a :
            for idx in b :
                if np.all(np.isin(idx0, idx)):
                    break
            else :
                return False
        return True
    #
    if not issubset(old, new) :
        sprint('Adaptive-Update : Different subsystems.')
        return True
    elif len(subkeys) == len(indices) :
        sprint('Adaptive-Keep : If bigger same subsystems.')
        return False
    #
    decompose = config[keysys]["decompose"]
    indices = decompose_sub(ions, **decompose)
    new = [np.asarray(x) for x in indices]
    if issubset(new, old) :
        if len(new) > len(old):
            sprint('Adaptive-Keep : Bigger will be contained.')
        elif len(new) == len(old):
            sprint('Adaptive-Keep : Same subsystems.')
        elif nsub_new_1 == len(old):
            sprint('Adaptive-Keep : Bigger will be same.')
        else :
            raise AttributeError("AdaptiveError : This shouldn't happened.")
        return False
    else :
        sprint('Adaptive-Update : If bigger will merge subsystems.')
        return True

def config2sub_global(config, ions, optimizer = None, grid = None, regions = None):
    keysys = "GSYSTEM"
    decompose = config[keysys]["decompose"]
    rtol = decompose.get('rtol', 0.0)
    if rtol > 1E-6 :
        update = config2sub_check_update(config, ions, rtol = rtol)
        if not update :
            reuse_drivers = {key : None for key in config if key.startswith('SUB')}
            return (config, reuse_drivers)

    if decompose['method'] == 'distance' :
        indices = decompose_sub(ions, **decompose)
    else :
        raise AttributeError("{} is not supported".format(decompose['method']))

    if grid is None :
        if optimizer is None :
            raise AttributeError("Please give the grid")
        grid = optimizer.grid

    # backup last step saved subs
    subkeys_prev = [key for key in config if key.startswith('SUB')]
    for key in subkeys_prev :
        config['B' + key] = config.pop(key)
    subkeys_prev = [key for key in config if key.startswith('BSUB')]
    reuse_drivers = {}

    asub = config['ASUB']
    subkeys = [key for key in asub if key.startswith('sub_')]

    hashes = OrderedDict()
    for i, index in enumerate(indices) :
        ind = set(index)
        this_hash = get_hash(ind)
        hashes[this_hash] = i
    # check saved keys
    for keysys in subkeys_prev :
        hash0 = config[keysys]['hash']
        if hash0 not in hashes : continue
        i = hashes[hash0]
        index = indices[i]
        this_key = keysys
        #the subsystem is same as last step
        key = this_key[1:]
        config[key] = config.pop(this_key)
        #-----------------------------------------------------------------------
        reuse_drivers[key] = None
        if optimizer is not None :
            for driver in optimizer.drivers :
                if driver is None : continue
                if driver.prefix == config[key]["prefix"] :
                    reuse_drivers[key] = driver
        #-----------------------------------------------------------------------
        indices[i] = []
        config[key]["density"]["file"] = None
        f_str = 'New Subsystem : {} {} {}'.format(key, 'saved', index)
        f_str = "\n".join(textwrap.wrap(f_str, width = 80))
        sprint(f_str)

    # check ASUB
    for i, index in enumerate(indices) :
        if len(index) == 0 : continue
        hs = OrderedDict()
        ind = set(index)
        nmax = 0
        na = 0
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
            basefiles = []
            for item, nc in hs.items() :
                nprocs += asub[item]["nprocs"]*nc/len(asub[item]["cell"]["index"])
                basefile = asub[item]["basefile"]
                if basefile : basefiles.append(basefile)
            config[key]["basefile"] = basefiles if len(basefiles)>1 else basefiles[0]
            config[key]["nprocs"] = max(1, nprocs)
            config[key]["cell"]["index"] = index
            #-----------------------------------------------------------------------
            if regions is None :
                regions = config2asub_region(config, ions, grid)
            lb = reduce(np.minimum, [regions[k][0] for k in hs])
            ub = reduce(np.maximum, [regions[k][1] for k in hs])
            cellsplit = np.minimum((ub - lb)/grid.nrR, 1.0)
            config[key]["cell"]["split"] = cellsplit.tolist()
            #-----------------------------------------------------------------------
        else :
            raise AttributeError("ERROR : Not support this kind : {}".format(kind))
        # set the initial density as None
        config[key]["density"]["file"] = None
        f_str = 'New Subsystem : {} {} {}'.format(key, kind, index)
        f_str = "\n".join(textwrap.wrap(f_str, width = 80))
        sprint(f_str)

    subkeys = [key for key in config if key.startswith('SUB')]

    if len(reuse_drivers) != len(subkeys) :
        # removed last step saved but unused subs
        for key in subkeys_prev : config.pop(key, None)
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
