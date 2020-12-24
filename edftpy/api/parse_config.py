import numpy as np
import os
from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.formats import ase_io, io

from edftpy.config import read_conf

from edftpy.pseudopotential import LocalPP
from edftpy.kedf import KEDF
from edftpy.hartree import Hartree
from edftpy.xc import XC
from edftpy.optimizer import Optimization
from edftpy.evaluator import Evaluator, EnergyEvaluatorMix, EvaluatorOF, TotalEvaluator
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

def config2optimizer(config, ions = None, optimizer = None, graphtopo = None, **kwargs):
    if isinstance(config, dict):
        pass
    elif isinstance(config, str):
        config = read_conf(config)
    #-----------------------------------------------------------------------
    graphtopo = config2graphtopo(config, graphtopo = graphtopo)
    #-----------------------------------------------------------------------
    cell_change = None
    ############################## Gsystem ##############################
    keysys = "GSYSTEM"
    if ions is None :
        try :
            ions = io.read(
                config["PATH"]["cell"] +os.sep+ config[keysys]["cell"]["file"],
                format=config[keysys]["cell"]["format"],
                names=config[keysys]["cell"]["elename"])
        except Exception:
            ions = ase_io.ase_read(config["PATH"]["cell"] +os.sep+ config[keysys]["cell"]["file"])
    elif optimizer is not None :
        if not np.allclose(optimizer.gsystem.ions.pos.cell.lattice, ions.pos.cell.lattice):
            cell_change = 'cell'
            raise AttributeError("Not support cell change")
        else :
            cell_change = 'position'
    nr = config[keysys]["grid"]["nr"]
    spacing = config[keysys]["grid"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
    ecut = config[keysys]["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    full=config[keysys]["grid"]["gfull"]
    max_prime = config[keysys]["grid"]["maxprime"]
    grid_scale = config[keysys]["grid"]["scale"]
    optfft = config[keysys]["grid"]["optfft"]
    pplist = {}
    for key in config["PP"]:
        ele = key.capitalize()
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
    total_evaluator = config2total_evaluator(config, ions, grid, pplist = pplist, total_evaluator=total_evaluator, cell_change = cell_change)
    gsystem.total_evaluator = total_evaluator
    ############################## Subsytem ##############################
    subkeys = []
    for key in config :
        if key.startswith('SUB'):
            subkeys.append(key)
    # subkeys.sort(reverse = True)
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
            if 'PSEUDO' in driver.evaluator.embed_evaluator.funcdicts :
                driver.evaluator.embed_evaluator.update_functional(remove = ['PSEUDO'])
                gsystem.total_evaluator.update_functional(remove = ['PSEUDO'])
            #-----------------------------------------------------------------------
        drivers.append(driver)
    #-----------------------------------------------------------------------
    # print('build_region -> ', graphtopo.rank)
    graphtopo.build_region(grid=gsystem.grid, drivers=drivers)
    #-----------------------------------------------------------------------
    for i, driver in enumerate(drivers):
        if driver is None : continue
        if (driver.technique == 'OF' and graphtopo.is_root) or (graphtopo.isub == i and graphtopo.comm_sub.rank == 0) or graphtopo.isub is None:
            # outfile = 'edftpy_subcell_' + str(i) + '.vasp'
            outfile = driver.prefix + '.vasp'
            ase_io.ase_write(outfile, driver.subcell.ions, format = 'vasp', direct = 'True', vasp5 = True, parallel = False)
    if graphtopo.is_root :
        ase_io.ase_write('edftpy_cell.vasp', ions, format = 'vasp', direct = 'True', vasp5 = True, parallel = False)
    #-----------------------------------------------------------------------
    optimization_options = config["OPT"].copy()
    optimization_options["econv"] *= ions.nat
    opt = Optimization(drivers = drivers, options = optimization_options, gsystem = gsystem)
    return opt

def config2graphtopo(config, subkeys = None, graphtopo = None):
    if graphtopo is None :
        graphtopo = GraphTopo()
    elif graphtopo.comm_sub is not None :
        # already initialize the comm_sub
        return graphtopo

    if subkeys is None :
        subkeys = []
        for key in config :
            if key.startswith('SUB'):
                subkeys.append(key)
    nprocs = []
    #OF driver set the procs to 0, make sure it use all resources
    for key in subkeys :
        if config[key]["technique"] == 'OF' :
            n = 0
        else :
            n = config[key]["nprocs"]
        nprocs.append(n)
    graphtopo.distribute_procs(nprocs)
    return graphtopo

def config2total_evaluator(config, ions, grid, pplist = None, total_evaluator= None, cell_change = None):
    keysys = "GSYSTEM"
    pme = config["MATH"]["linearie"]
    xc_kwargs = config[keysys]["exc"].copy()
    ke_kwargs = config[keysys]["kedf"].copy()
    #---------------------------Functional----------------------------------
    if cell_change == 'position' :
        pseudo = total_evaluator.funcdicts['PSEUDO']
        pseudo.restart(grid=grid, ions=ions, full=False)
        total_evaluator.funcdicts['PSEUDO'] = pseudo
    else :
        pseudo = LocalPP(grid = grid, ions=ions, PP_list=pplist, PME=pme)
        hartree = Hartree()
        xc = XC(**xc_kwargs)
        funcdicts = {'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
        if ke_kwargs['kedf'] == 'None' :
            pass
        else :
            ke = KEDF(**ke_kwargs)
            funcdicts['KE'] = ke
        total_evaluator = TotalEvaluator(**funcdicts)
    return total_evaluator

def config2evaluator(config, keysys, ions, grid, pplist = None, optimizer = None, cell_change = None):
    emb_ke_kwargs = config['GSYSTEM']["kedf"].copy()
    emb_xc_kwargs = config['GSYSTEM']["exc"].copy()
    pme = config["MATH"]["linearie"]

    ke_kwargs = config[keysys]["kedf"].copy()
    xc_kwargs = config[keysys]["exc"].copy()
    embed = config[keysys]["embed"]
    technique = config[keysys]["technique"]
    exttype = config[keysys]["exttype"]
    #Embedding Functional---------------------------------------------------
    if exttype is not None :
        embed = ['KE']
        if not exttype & 1 : embed.append('PSEUDO')
        if not exttype & 2 : embed.append('HARTREE')
        if not exttype & 4 : embed.append('XC')
        # if exttype == 0 : embed = []
    emb_funcdicts = {}
    if 'KE' in embed :
        ke_emb = KEDF(**emb_ke_kwargs)
        emb_funcdicts['KE'] = ke_emb
    if 'XC' in embed :
        xc_emb = XC(**emb_xc_kwargs)
        emb_funcdicts['XC'] = xc_emb
    if 'HARTREE' in embed :
        hartree = Hartree()
        emb_funcdicts['HARTREE'] = hartree
    if 'PSEUDO' in embed :
        pseudo = LocalPP(grid = grid, ions=ions,PP_list=pplist,PME=pme)
        emb_funcdicts['PSEUDO'] = pseudo
    emb_evaluator = Evaluator(**emb_funcdicts)
    #Subsystem Functional---------------------------------------------------
    # ke_sub_kwargs = {'name' :'vW'}
    # ke_sub = KEDF(**ke_sub_kwargs)
    # sub_funcdicts = {'KE' :ke_sub}
    sub_funcdicts = {}
    if 'XC' in embed :
        xc_sub = XC(**xc_kwargs)
        sub_funcdicts['XC'] = xc_sub
    sub_evaluator = Evaluator(**sub_funcdicts)
    # Remove the vW part from the KE
    ke_kwargs['y'] = 0.0
    ke_evaluator = KEDF(**ke_kwargs)
    #KS not need sub_evaluator and ke_evaluator-----------------------------
    if technique == 'KS' :
        sub_evaluator = None
        ke_evaluator = None
    return (emb_evaluator, sub_evaluator, ke_evaluator)

def config2driver(config, keysys, ions, grid, pplist = None, optimizer = None, cell_change = None, driver = None, mp = None, comm = None):
    gsystem_ecut = config['GSYSTEM']["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    full=config['GSYSTEM']["grid"]["gfull"]
    pp_path = config["PATH"]["pp"]
    max_prime_global = config['GSYSTEM']["grid"]["maxprime"]
    grid_scale_global = config['GSYSTEM']["grid"]["scale"]

    # nr = config[keysys]["grid"]["nr"]
    # spacing = config[keysys]["grid"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
    # print('keysys', keysys, config[keysys])
    # print_conf(config[keysys])

    ecut = config[keysys]["grid"]["ecut"]
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
    if atomicfiles :
        for k, v in atomicfiles.copy().items():
            if not os.path.exists(v):
                atomicfiles[k] = pp_path + os.sep + v
    if not prefix :
        prefix = keysys.lower()
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
        for key in set(ions.labels):
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
    subcell = SubCell(ions, grid, index = index, cellcut = cellcut, cellsplit = cellsplit, optfft = True, gaussian_options = gaussian_options, grid_sub = grid_sub, max_prime = max_prime, scale = grid_scale, mp = mp)

    if cell_change == 'position' :
        if subcell.density.shape == driver.density.shape :
            subcell.density[:] = driver.density
    else :
        if infile : # initial='Read'
            subcell.density[:] = io.read_density(infile)
        elif initial == 'Atomic' and len(atomicfiles) > 0 :
            atomicd = AtomicDensity(files = atomicfiles)
            subcell.density[:] = atomicd.guess_rho(subcell.ions, subcell.grid)
        elif initial or technique == 'OF' :
            atomicd = AtomicDensity()
            subcell.density[:] = atomicd.guess_rho(subcell.ions, subcell.grid)
            subcell.density[:] = dftpy_opt(subcell.ions, subcell.density, pplist)
    #-----------------------------------------------------------------------
    if mix_kwargs['scheme'] == 'Pulay' :
        mixer = PulayMixer(**mix_kwargs)
    elif mix_kwargs['scheme'] == 'Linear' :
        mixer = LinearMixer(**mix_kwargs)
    elif mix_kwargs['scheme'].capitalize() in ['No', 'None'] :
        mixer = mix_kwargs['coef'][0]
    else :
        raise AttributeError("!!!ERROR : NOT support ", mix_kwargs['scheme'])
    #-----------------------------------------------------------------------
    embed_evaluator, sub_evaluator, ke_evaluator = config2evaluator(config, keysys, subcell.ions, subcell.grid, pplist = pplist, optimizer = optimizer, cell_change = cell_change)
    ke_sub_kwargs = {'name' :'vW'}
    ke_sub = KEDF(**ke_sub_kwargs)

    exttype = 7
    if embed_evaluator is not None :
        if 'XC' in embed_evaluator.funcdicts :
            exttype -= 4
        if 'HARTREE' in embed_evaluator.funcdicts :
            exttype -= 2
        if 'PSEUDO' in embed_evaluator.funcdicts :
            exttype -= 1

    def get_dftpy_driver(energy_evaluator):
        opt_options['econv'] *= subcell.ions.nat
        if ecut and abs(ecut - gsystem_ecut) > 1.0 :
            if comm.size > 1 :
                raise AttributeError("Different energy cutoff not supported for parallel version")
            if cell_change == 'position' :
                total_evaluator = optimizer.evaluator_of.gsystem.total_evaluator
                gsystem_driver = optimizer.evaluator_of.gsystem
            else :
                total_evaluator = None
                gsystem_driver = GlobalCell(ions, grid = None, ecut = ecut, full = full, optfft = True, max_prime = max_prime_global, scale = grid_scale_global, mp = mp)
            total_evaluator = config2total_evaluator(config, ions, gsystem_driver.grid, pplist = pplist, total_evaluator=total_evaluator, cell_change = cell_change)
            gsystem_driver.total_evaluator = total_evaluator
            grid_sub = gsystem_driver.grid
            grid_sub.shift = np.zeros(3, dtype = 'int32')
        else :
            grid_sub = None
            gsystem_driver = None

        add_ke = {}
        if opt_options['opt_method'] == 'full' :
            if ke_sub is not None : add_ke = {'KE' : ke_sub}
        else :
            if ke_evaluator is not None : add_ke = {'KE' :ke_evaluator}
        sub_eval = sub_evaluator
        if sub_eval is None :
            sub_eval = Evaluator(**add_ke)
        else :
            sub_eval.update_functional(add = add_ke)

        if opt_options['opt_method'] == 'full' :
            mixer_of = LinearMixer(predtype = None, coef = [1.0], predecut = None, delay = 1)
            add_ke = {'KE' : ke_sub}
            sub_eval = sub_evaluator
            if sub_eval is None :
                sub_eval = Evaluator(**add_ke)
            else :
                sub_eval.update_functional(add = add_ke)
            evaluator_of = EvaluatorOF(sub_evaluator = sub_eval, gsystem = gsystem_driver)
        else :
            evaluator_of = EvaluatorOF(sub_evaluator = sub_eval, gsystem = gsystem_driver, ke_evaluator = ke_sub)
            mixer_of = mixer
        driver = DFTpyOF(evaluator =energy_evaluator, prefix = prefix, options = opt_options, subcell = subcell,
                mixer = mixer_of, evaluator_of = evaluator_of, grid = grid_sub)
        return driver

    energy_evaluator = EnergyEvaluatorMix(embed_evaluator = embed_evaluator)
    if calculator == 'dftpy' and opt_options['opt_method'] == 'full' :
        energy_evaluator = EnergyEvaluatorMix(embed_evaluator = embed_evaluator, ke_evaluator = ke_evaluator)

    margs = {
            'evaluator' :energy_evaluator,
            'prefix' : prefix,
            'subcell' : subcell,
            'cell_params' : None,
            'params' : None,
            'exttype' : exttype,
            'base_in_file' : basefile,
            'mixer' : mixer,
            'comm' : mp.comm
            }
    if cell_change == 'cell' :
        raise AttributeError("Not support change the cell")

    if cell_change == 'position' :
        driver.update_workspace(subcell)
    else :
        if calculator == 'dftpy' :
            driver = get_dftpy_driver(energy_evaluator)
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
        if abs(ecut - gsystem_ecut) < 1.0 :
            params['devel_code'] = 'STD_GRID={0} {1} {2}  FINE_GRID={0} {1} {2}'.format(*subcell.grid.nr)

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
        params['system']['ecutwfc'] = ecut * 2.0
        if abs(4 * ecut - gsystem_ecut) < 1.0 :
            params['system']['nr1'] = subcell.grid.nr[0]
            params['system']['nr2'] = subcell.grid.nr[1]
            params['system']['nr3'] = subcell.grid.nr[2]

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
