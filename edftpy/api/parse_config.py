import numpy as np
import os
from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.formats import ase_io, io

from edftpy.config import read_conf, print_conf

from edftpy.pseudopotential import LocalPP
from edftpy.kedf import KEDF
from edftpy.hartree import Hartree
from edftpy.xc import XC
from edftpy.optimizer import Optimization
from edftpy.evaluator import Evaluator, EnergyEvaluatorMix, EvaluatorOF
from edftpy.enginer.driver import OptDriver
from edftpy.enginer.of_dftpy import DFTpyOF
from edftpy.density.init_density import AtomicDensity
from edftpy.subsystem.subcell import SubCell, GlobalCell
from edftpy.mixer import LinearMixer, PulayMixer
from edftpy.enginer.of_dftpy import dftpy_opt

def config2optimizer(config, ions = None, optimizer = None, **kwargs):
    if isinstance(config, dict):
        pass
    elif isinstance(config, str):
        config = read_conf(config)
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
        else :
            cell_change = 'position'
    nr = config[keysys]["grid"]["nr"]
    spacing = config[keysys]["grid"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
    ecut = config[keysys]["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    full=config[keysys]["grid"]["gfull"]
    max_prime = config[keysys]["grid"]["maxprime"]
    grid_scale = config[keysys]["grid"]["scale"]
    pplist = {}
    for key in config["PP"]:
        ele = key.capitalize()
        pplist[ele] = config["PATH"]["pp"] +os.sep+ config["PP"][key]
    #---------------------------Functional----------------------------------
    if cell_change == 'position' :
        total_evaluator = optimizer.gsystem.total_evaluator
        gsystem = optimizer.gsystem
        gsystem.restart(grid=gsystem.grid, ions=ions)
    else :
        total_evaluator = None
        gsystem = GlobalCell(ions, grid = None, ecut = ecut, nr = nr, spacing = spacing, full = full, optfft = True, max_prime = max_prime, scale = grid_scale)
    grid = gsystem.grid
    total_evaluator = config2total_evaluator(config, ions, grid, pplist = pplist, total_evaluator=total_evaluator, cell_change = cell_change)
    gsystem.total_evaluator = total_evaluator
    ############################## Subsytem ##############################
    subkeys = []
    for key in config :
        if key.startswith('SUB'):
            subkeys.append(key)
    subkeys.sort()
    # subkeys.sort(reverse = True)
    opt_drivers = []
    ns = 0
    for keysys in subkeys :
        if cell_change == 'position' :
            driver = optimizer.opt_drivers[ns]
        else :
            driver = None
        driver = config2driver(config, keysys, ions, grid, pplist, optimizer = optimizer, cell_change = cell_change, driver = driver)
        opt_drivers.append(driver)
        ns += 1
    #-----------------------------------------------------------------------
    for i in range(len(opt_drivers)):
        ase_io.ase_write('edftpy_subcell_' + str(i) + '.vasp', opt_drivers[i].calculator.subcell.ions, format = 'vasp', direct = 'True', vasp5 = True)
    ase_io.ase_write('edftpy_cell.vasp', ions, format = 'vasp', direct = 'True', vasp5 = True)
    #-----------------------------------------------------------------------
    optimization_options = config["OPT"].copy()
    optimization_options["econv"] *= ions.nat
    opt = Optimization(opt_drivers = opt_drivers, options = optimization_options, gsystem = gsystem)
    return opt

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
        ke = KEDF(**ke_kwargs)
        funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
        total_evaluator = Evaluator(**funcdicts)
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

def config2driver(config, keysys, ions, grid, pplist = None, optimizer = None, cell_change = None, driver = None):
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
        grid_sub = driver.calculator.subcell.grid
    else :
        grid_sub = None
    subsys = SubCell(ions, grid, index = index, cellcut = cellcut, cellsplit = cellsplit, optfft = True, gaussian_options = gaussian_options, grid_sub = grid_sub, max_prime = max_prime, scale = grid_scale)

    if cell_change == 'position' :
        subsys.density[:] = driver.calculator.subcell.density
    else :
        if infile : # initial='Read'
            subsys.density[:] = io.read_density(infile)
        elif initial == 'Atomic' and len(atomicfiles) > 0 :
            atomicd = AtomicDensity(files = atomicfiles)
            subsys.density[:] = atomicd.guess_rho(subsys.ions, subsys.grid)
        elif initial or technique == 'OF' :
            atomicd = AtomicDensity()
            subsys.density[:] = atomicd.guess_rho(subsys.ions, subsys.grid)
            dftpy_opt(subsys.ions, subsys.density, pplist)
    #-----------------------------------------------------------------------
    if mix_kwargs['scheme'] == 'Pulay' :
        mixer = PulayMixer(**mix_kwargs)
    elif mix_kwargs['scheme'] == 'Linear' :
        mixer = LinearMixer(**mix_kwargs)
    else :
        raise AttributeError("!!!ERROR : NOT support ", mix_kwargs['scheme'])
    #-----------------------------------------------------------------------
    embed_evaluator, sub_evaluator, ke_evaluator = config2evaluator(config, keysys, subsys.ions, subsys.grid, pplist = pplist, optimizer = optimizer, cell_change = cell_change)
    ke_sub_kwargs = {'name' :'vW'}
    ke_sub = KEDF(**ke_sub_kwargs)

    exttype = 7
    if 'XC' in embed_evaluator.funcdicts :
        exttype -= 4
    if 'HARTREE' in embed_evaluator.funcdicts :
        exttype -= 2
    if 'PSEUDO' in embed_evaluator.funcdicts :
        exttype -= 1
    print('exttype', exttype)

    def get_dftpy_enginer():
        opt_options['econv'] *= subsys.ions.nat

        if ecut and abs(ecut - gsystem_ecut) > 1.0 :
            if cell_change == 'position' :
                total_evaluator = optimizer.evaluator_of.gsystem.total_evaluator
                gsystem_driver = optimizer.evaluator_of.gsystem
            else :
                total_evaluator = None
                gsystem_driver = GlobalCell(ions, grid = None, ecut = ecut, full = full, optfft = True, max_prime = max_prime_global, scale = grid_scale_global)
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
        enginer = DFTpyOF(options = opt_options, subcell = subsys, mixer = mixer_of, evaluator_of = evaluator_of, grid = grid_sub)
        return enginer

    def get_castep_enginer():
        from edftpy.enginer.ks_castep import CastepKS
        cell_params = {'species_pot' : pplist}
        if kpoints['grid'] is not None :
            cell_params['kpoints_mp_grid'] = ' '.join(map(str, kpoints['grid']))

        params = {}
        if ecut :
            params['cut_off_energy'] = ecut * ENERGY_CONV['Hartree']['eV']
            if abs(ecut - gsystem_ecut) < 1.0 :
                params['devel_code'] = 'STD_GRID={0} {1} {2}  FINE_GRID={0} {1} {2}'.format(*subsys.grid.nr)

        enginer = CastepKS(prefix = prefix, subcell = subsys, cell_params = cell_params, params = params, exttype = exttype,
                base_in_file = basefile, mixer = mixer)
        return enginer

    def get_pwscf_enginer():
        from edftpy.enginer.ks_pwscf import PwscfKS

        cell_params = {'pseudopotentials' : pplist}
        if kpoints['grid'] is not None :
            cell_params['kpts'] = kpoints['grid']
        if kpoints['offset'] is not None :
            cell_params['koffset'] = kpoints['grid']

        params = {'system' : {}}
        if ecut :
            params['system']['ecutwfc'] = ecut * 2.0
            if abs(4 * ecut - gsystem_ecut) < 1.0 :
                params['system']['nr1'] = subsys.grid.nr[0]
                params['system']['nr2'] = subsys.grid.nr[1]
                params['system']['nr3'] = subsys.grid.nr[2]

        enginer = PwscfKS(prefix = prefix, subcell = subsys, cell_params = cell_params, params = params, exttype = exttype,
                base_in_file = basefile, mixer = mixer)
        return enginer

    if calculator == 'dftpy' :
        enginer = get_dftpy_enginer()
    elif calculator == 'pwscf' :
        enginer = get_pwscf_enginer()
    elif calculator == 'castep' :
        enginer = get_castep_enginer()

    energy_evaluator = EnergyEvaluatorMix(embed_evaluator = embed_evaluator)
    if calculator == 'dftpy' and opt_options['opt_method'] == 'full' :
        energy_evaluator = EnergyEvaluatorMix(embed_evaluator = embed_evaluator, ke_evaluator = ke_evaluator)

    driver = OptDriver(energy_evaluator = energy_evaluator, calculator = enginer)
    return driver
