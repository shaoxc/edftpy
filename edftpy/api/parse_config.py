import numpy as np
import os
from collections import OrderedDict
import copy

from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.formats import ase_io, io

from edftpy.config import read_conf, print_conf, write_conf

from edftpy.pseudopotential import LocalPP
from edftpy.kedf import KEDF
from edftpy.hartree import Hartree
from edftpy.xc import XC
from edftpy.optimizer import Optimization
from edftpy.evaluator import EmbedEvaluator, EvaluatorOF, TotalEvaluator
from edftpy.density.init_density import AtomicDensity
from edftpy.subsystem.subcell import SubCell, GlobalCell
from edftpy.mixer import LinearMixer, PulayMixer
from edftpy.enginer.of_dftpy import DFTpyOF, dftpy_opt
from edftpy.mpi import GraphTopo, MP, sprint

def import_drivers(config):
    """
    Import the enginer of different drivers

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
    if 'pwscf' in calcs :
        from edftpy.enginer.ks_pwscf import PwscfKS
    if 'castep' in calcs :
        from edftpy.enginer.ks_castep import CastepKS
    if 'dftpy' in calcs :
        from edftpy.enginer.of_dftpy import DFTpyOF
    return

def config_correct(config):
    """

    Correct some missing but important variables

    Args:
        config: config

    Notes:
        Only correct some missing variables
    """
    subkeys = [key for key in config if key.startswith('SUB')]
    for key in subkeys :
        if config[key]['calculator'] == 'dftpy' :
            config[key]['technique'] = 'OF'
        else :
            config[key]['technique'] = 'KS'

        if not config[key]["prefix"] :
            config[key]["prefix"] = key.lower()

        if config[key]['density']['output'] :
            if config[key]['density']['output'].startswith('.') :
                config[key]['density']['output'] = config[key]["prefix"] + config[key]['density']['output']

    return config

def config2optimizer(config, ions = None, optimizer = None, graphtopo = None, pseudo = None, **kwargs):
    if isinstance(config, dict):
        pass
    elif isinstance(config, str):
        config = read_conf(config)
    ############################## Gsystem ##############################
    cell_change = None
    keysys = "GSYSTEM"
    if ions is None :
        try :
            ions = io.read(
                config["PATH"]["cell"] +os.sep+ config[keysys]["cell"]["file"],
                format=config[keysys]["cell"]["format"],
                names=config[keysys]["cell"]["elename"])
        except Exception:
            ions = ase_io.ase_read(config["PATH"]["cell"] +os.sep+ config[keysys]["cell"]["file"])

    if optimizer is not None :
        if not np.allclose(optimizer.gsystem.ions.pos.cell.lattice, ions.pos.cell.lattice):
            cell_change = None # cell_change = 'cell'
            # except PSEUDO, clean everything
            for i, driver in enumerate(optimizer.drivers):
                if driver is None : continue
                driver.stop_run()
            pseudo = optimizer.gsystem.total_evaluator.funcdicts['PSEUDO']
        else :
            cell_change = 'position'

    config = config2nsub(config, ions)
    #-----------------------------------------------------------------------
    config = config_correct(config)
    #-----------------------------------------------------------------------
    graphtopo = config2graphtopo(config, graphtopo = graphtopo)
    if graphtopo.rank == 0 :
        config_json = config_to_json(config, ions)
        write_conf('eDFTpy_running.json', config_json)
    #-----------------------------------------------------------------------
    nr = config[keysys]["grid"]["nr"]
    spacing = config[keysys]["grid"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
    ecut = config[keysys]["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    full=config[keysys]["grid"]["gfull"]
    max_prime = config[keysys]["grid"]["maxprime"]
    grid_scale = config[keysys]["grid"]["scale"]
    optfft = config[keysys]["grid"]["optfft"]
    labels = set(ions.labels)
    pplist = {}
    for key in config["PP"]:
        ele = key.capitalize()
        if ele in labels :
            pplist[ele] = config["PATH"]["pp"] +os.sep+ config["PP"][key]
    #---------------------------Functional----------------------------------
    if cell_change == 'position' :
        total_evaluator = optimizer.gsystem.total_evaluator
        gsystem = optimizer.gsystem
        gsystem.restart(grid=gsystem.grid, ions=ions)
        mp_global = gsystem.grid.mp
    else :
        total_evaluator = None
        mp_global = MP(comm = graphtopo.comm, parallel = graphtopo.is_mpi)
        gsystem = GlobalCell(ions, grid = None, ecut = ecut, nr = nr, spacing = spacing, full = full, optfft = optfft, max_prime = max_prime, scale = grid_scale, mp = mp_global, graphtopo = graphtopo)
    grid = gsystem.grid
    total_evaluator = config2total_evaluator(config, ions, grid, pplist = pplist, total_evaluator=total_evaluator, cell_change = cell_change, pseudo = pseudo)
    gsystem.total_evaluator = total_evaluator
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
                mp = mp_global
            else :
                mp = MP(comm = graphtopo.comm_sub)
            driver = config2driver(config, keysys, ions, grid, pplist, optimizer = optimizer, cell_change = cell_change, driver = driver, mp = mp)
            #-----------------------------------------------------------------------
            #PSEUDO was evaluated on all processors, so directly remove from embedding
            # if 'PSEUDO' in driver.evaluator.funcdicts :
            #     driver.evaluator.update_functional(remove = ['PSEUDO'])
            #     gsystem.total_evaluator.update_functional(remove = ['PSEUDO'])
            #-----------------------------------------------------------------------
        drivers.append(driver)
    #-----------------------------------------------------------------------
    # sprint('build_region -> ', graphtopo.rank)
    graphtopo.build_region(grid=gsystem.grid, drivers=drivers)
    #-----------------------------------------------------------------------
    for i, driver in enumerate(drivers):
        if driver is None : continue
        if (driver.technique == 'OF' and graphtopo.is_root) or (graphtopo.isub == i and graphtopo.comm_sub.rank == 0) or graphtopo.isub is None:
            outfile = driver.prefix + '.vasp'
            ase_io.ase_write(outfile, driver.subcell.ions, format = 'vasp', direct = 'True', vasp5 = True, parallel = False)
            # io.write(driver.prefix +'.xsf', driver.density, driver.subcell.ions)
    if graphtopo.is_root :
        ase_io.ase_write('edftpy_cell.vasp', ions, format = 'vasp', direct = 'True', vasp5 = True, parallel = False)
    #-----------------------------------------------------------------------
    # io.write('total.xsf', gsystem.density, gsystem.ions)
    # graphtopo.comm.Barrier()
    optimization_options = config["OPT"].copy()
    optimization_options["econv"] *= ions.nat
    opt = Optimization(drivers = drivers, options = optimization_options, gsystem = gsystem)
    return opt

def config2graphtopo(config, graphtopo = None):
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
        graphtopo.distribute_procs(nprocs)
        sprint('Communicators recreated : ', graphtopo.comm.size, comm = graphtopo.comm)
    else :
        sprint('Communicators already created : ', graphtopo.comm.size, comm = graphtopo.comm)
    sprint('Number of subsystems : ', len(graphtopo.nprocs), comm = graphtopo.comm)
    f_str = np.array2string(graphtopo.nprocs, separator=' ', max_line_width=80)
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
            # pseudo(calcType = ['V'])
            # exit()
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
        ke_evaluator = KEDF(**ke_kwargs)
        sub_funcdicts['KE'] = ke_evaluator
        evaluator_of = EvaluatorOF(gsystem = gsystem, ke_evaluator = ke_sub, **sub_funcdicts)

    return evaluator_of

def config2driver(config, keysys, ions, grid, pplist = None, optimizer = None, cell_change = None, driver = None, mp = None, comm = None):
    gsystem_ecut = config['GSYSTEM']["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    full=config['GSYSTEM']["grid"]["gfull"]
    pp_path = config["PATH"]["pp"]
    max_prime_global = config['GSYSTEM']["grid"]["maxprime"]
    grid_scale_global = config['GSYSTEM']["grid"]["scale"]
    optfft_global = config['GSYSTEM']["grid"]["optfft"]

    # print('keysys', keysys, config[keysys])
    # print_conf(config[keysys])

    ecut = config[keysys]["grid"]["ecut"]
    nr = config[keysys]["grid"]["nr"]
    max_prime = config[keysys]["grid"]["maxprime"]
    grid_scale = config[keysys]["grid"]["scale"]
    cellcut = config[keysys]["cell"]["cut"]
    cellsplit= config[keysys]["cell"]["split"]
    index = config[keysys]["cell"]["index"]
    initial = config[keysys]["density"]["initial"]
    infile = config[keysys]["density"]["file"]
    use_gaussians = config[keysys]["density"]["use_gaussians"]
    gaussians_rcut = config[keysys]["density"]["gaussians_rcut"]
    gaussians_sigma = config[keysys]["density"]["gaussians_sigma"]
    gaussians_scale = config[keysys]["density"]["gaussians_scale"]
    technique = config[keysys]["technique"]
    opt_options = config[keysys]["opt"].copy()
    calculator = config[keysys]["calculator"]
    mix_kwargs = config[keysys]["mix"].copy()
    basefile = config[keysys]["basefile"]
    prefix = config[keysys]["prefix"]
    kpoints = config[keysys]["kpoints"]
    exttype = config[keysys]["exttype"]
    atomicfiles = config[keysys]["density"]["atomic"].copy()
    optfft = config[keysys]["grid"]["optfft"]
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
    subcell = SubCell(ions, grid, index = index, cellcut = cellcut, cellsplit = cellsplit, optfft = optfft, gaussian_options = gaussian_options, grid_sub = grid_sub, max_prime = max_prime, scale = grid_scale, nr = nr, mp = mp)

    if cell_change == 'position' :
        if subcell.density.shape == driver.density.shape :
            subcell.density[:] = driver.density
    else :
        if infile : # initial='Read'
            subcell.density[:] = io.read_density(infile)
        elif initial == 'Atomic' and len(atomicfiles) > 0 :
            atomicd = AtomicDensity(files = atomicfiles)
            subcell.density[:] = atomicd.guess_rho(subcell.ions, subcell.grid)
        elif initial == 'Heg' :
            atomicd = AtomicDensity()
            subcell.density[:] = atomicd.guess_rho(subcell.ions, subcell.grid)
        elif initial or technique == 'OF' :
            atomicd = AtomicDensity()
            subcell.density[:] = atomicd.guess_rho(subcell.ions, subcell.grid)
            subcell.density[:] = dftpy_opt(subcell.ions, subcell.density, pplist)
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

    def get_dftpy_driver(pplist, gsystem_ecut = None, ecut = None, kpoints = {}, margs = {}, **kwargs):
        subcell = margs.get('subcell')
        opt_options['econv'] *= subcell.ions.nat
        if ecut and abs(ecut - gsystem_ecut) > 1.0 :
            if comm.size > 1 :
                raise AttributeError("Different energy cutoff not supported for parallel version")
            if cell_change == 'position' :
                total_evaluator = optimizer.evaluator_of.gsystem.total_evaluator
                gsystem_driver = optimizer.evaluator_of.gsystem
            else :
                total_evaluator = None
                gsystem_driver = GlobalCell(ions, grid = None, ecut = ecut, full = full, optfft = optfft_global, max_prime = max_prime_global, scale = grid_scale_global, mp = mp)
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
        driver = DFTpyOF(**margs)
        return driver

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
            'ncharge' : ncharge
            }

    if cell_change == 'cell' :
        raise AttributeError("Not support change the cell")

    if cell_change == 'position' :
        driver.update_workspace(subcell)
    else :
        if calculator == 'dftpy' :
            driver = get_dftpy_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)
        elif calculator == 'pwscf' :
            driver = get_pwscf_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)
        elif calculator == 'castep' :
            driver = get_castep_driver(pplist, gsystem_ecut = gsystem_ecut, ecut = ecut, kpoints = kpoints, margs = margs)

    return driver

def get_castep_driver(pplist, gsystem_ecut = None, ecut = None, kpoints = {}, margs = {}, **kwargs):
    from edftpy.driver.ks_castep import CastepKS
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
    driver = CastepKS(**margs)
    # driver = CastepKS(evaluator =energy_evaluator, prefix = prefix, subcell = subcell, cell_params = cell_params,
    # params = params, exttype = exttype, base_in_file = basefile, mixer = mixer, comm = mp.comm)
    return driver

def get_pwscf_driver(pplist, gsystem_ecut = None, ecut = None, kpoints = {}, margs = {}, **kwargs):
    from edftpy.enginer.ks_pwscf import PwscfKS

    cell_params = {'pseudopotentials' : pplist}
    if kpoints['grid'] is not None :
        cell_params['kpts'] = kpoints['grid']
    if kpoints['offset'] is not None :
        cell_params['koffset'] = kpoints['grid']

    params = {'system' : {}}
    if ecut :
        subcell = margs['subcell']
        # params['system']['ecutwfc'] = ecut * 2.0
        # if abs(4 * ecut - gsystem_ecut) < 1.0 :
        params['system']['ecutwfc'] = ecut / 4.0 * 2.0 # 4*ecutwfc to Ry
        if abs(ecut - gsystem_ecut) < 5.0 :
            params['system']['nr1'] = subcell.grid.nrR[0]
            params['system']['nr2'] = subcell.grid.nrR[1]
            params['system']['nr3'] = subcell.grid.nrR[2]

    add = {
            'cell_params': cell_params,
            'params': params,
            }
    margs.update(add)
    driver = PwscfKS(**margs)
    return driver

def _get_gap(config, optimizer):
    for key in config :
        if key.startswith('SUB'):
            config[key]['opt']['opt_method'] = 'hamiltonian'
    config['OPT']['maxiter'] = 10
    optimizer = config2optimizer(config, optimizer.gsystem.ions, optimizer)
    optimizer.optimize()

def config2nsub(config, ions):
    from edftpy.subsystem.decompose import from_distance_to_sub
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
        decompose = config[keysys]["decompose"]
        if decompose['method'] != 'distance' :
            raise AttributeError("{} is not supported".format(decompose['method']))
        index = config[keysys]["cell"]["index"]
        ions_sub = ions[index]

        radius = decompose['radius']
        if len(radius) == 0 :
            cutoff = decompose['rcut']
        else :
            keys = list(radius.keys())
            if not set(keys) >= set(list(ions_sub.nsymbols)) :
                raise AttributeError("The radius should contains all the elements")
            cutoff = {}
            for i, k in enumerate(keys):
                for k1 in keys[i:] :
                    cutoff[(k, k1)] = radius[k] + radius[k1]

        if decompose['method'] == 'distance' :
            indices = from_distance_to_sub(ions_sub, cutoff = cutoff)

        ions_inds_sub = ions_inds[index]
        for i, ind in enumerate(indices) :
            indices[i] = ions_inds_sub[ind]

        config = config_from_index(config, keysys, indices)
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
        config[key]['decompose']['adaptive'] = 'manual'
        config[key]['cell']['index'] = ind
        config[key]["nprocs"] = max(1, config[keysys]["nprocs"] // len(indices))
        config[key]['subs'] = None
        subs.append(key)
    config[keysys]['subs'] = subs
    if config[keysys]['decompose']['adaptive'] == 'manual' :
        del config[keysys]
    return config

def config_to_json(config, ions):
    """
    Note :
        fix some python types that not supported by json.dump
    """
    #-----------------------------------------------------------------------
    # slice and np.ndarray
    config_json = copy.deepcopy(config)
    # subkeys = [key for key in config_json if key.startswith('NSUB')]
    # for keysys in subkeys : del config_json[keysys]
    subkeys = [key for key in config_json if key.startswith(('SUB', 'NSUB'))]
    index_w = np.arange(0, ions.nat)
    for keysys in subkeys :
        index = config_json[keysys]["cell"]["index"]
        if isinstance(index, slice):
            index = index_w[index]
            config_json[keysys]["cell"]["index"] = index.tolist()
        elif isinstance(index, np.ndarray):
            config_json[keysys]["cell"]["index"] = index.tolist()
    #-----------------------------------------------------------------------
    return config_json
