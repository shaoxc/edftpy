import numpy as np
import time
import os
from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.formats import ase_io, io
from dftpy.ewald import ewald

from edftpy.config import read_conf, print_conf

from edftpy.utils.common import Field, Grid, Atoms
from edftpy.pseudopotential import LocalPP
from edftpy.kedf import KEDF
from edftpy.hartree import Hartree
from edftpy.xc import XC
from edftpy.optimizer import Optimization
from edftpy.evaluator import Evaluator, EnergyEvaluatorMix
from edftpy.enginer.driver import OptDriver
from edftpy.enginer.of_dftpy import DFTpyOF
from edftpy.density.init_density import AtomicDensity
from edftpy.subsystem.subcell import SubCell, GlobalCell
from edftpy.mixer import LinearMixer, PulayMixer

def optimize_density_conf(config, **kwargs):
    opt = config2optimizer(config)
    opt.optimize()
    rho = opt.density
    energy = opt.energy
    ions = opt.gsystem.ions
    ewald_ = ewald(rho=rho, ions=ions, PME=True)
    print('Final energy (a.u.)', energy + ewald_.energy)
    print('Final energy (eV)', (energy + ewald_.energy) * ENERGY_CONV['Hartree']['eV'])
    return

def config2optimizer(config, **kwargs):
    if isinstance(config, dict):
        pass
    elif isinstance(config, str):
        config = read_conf(config)
    print_conf(config)
    #-----------------------------------------------------------------------
    ions = None
    grid = None
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
    nr = config[keysys]["grid"]["nr"]
    spacing = config[keysys]["grid"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
    ecut = config[keysys]["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    full=config[keysys]["grid"]["gfull"]
    pplist = {}
    for key in config["PP"]:
        ele = key.capitalize()
        pplist[ele] = config["PATH"]["pp"] +os.sep+ config["PP"][key]
    pme = config["MATH"]["linearie"]
    xc_kwargs = config[keysys]["exc"]
    ke_kwargs = config[keysys]["kedf"]

    #---------------------------Functional----------------------------------
    gsystem = GlobalCell(ions, grid = grid, ecut = ecut, nr = nr, spacing = spacing, full = full, optfft = True)
    grid = gsystem.grid
    pseudo = LocalPP(grid = grid, ions=ions,PP_list=pplist,PME=pme)
    hartree = Hartree()
    xc = XC(**xc_kwargs)
    ke = KEDF(**ke_kwargs)
    funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
    total_evaluator = Evaluator(**funcdicts)
    gsystem.total_evaluator = total_evaluator
    ############################## Subsytem ##############################
    subkeys = []
    for key in config :
        if key.startswith('SUB'):
            subkeys.append(key)
    subkeys.sort()
    # subkeys.sort(reverse = True)
    opt_drivers = []
    for keysys in subkeys :
        driver = config2driver(config, keysys, ions, grid, pplist)
        opt_drivers.append(driver)
    #-----------------------------------------------------------------------
    for i in range(len(opt_drivers)):
        ase_io.ase_write('edftpy_subcell_' + str(i) + '.vasp', opt_drivers[i].calculator.subcell.ions, format = 'vasp', direct = 'True', vasp5 = True)
    #-----------------------------------------------------------------------
    optimization_options = config["OPT"]
    optimization_options["econv"] *= ions.nat
    opt = Optimization(opt_drivers = opt_drivers, options = optimization_options, gsystem = gsystem)
    return opt

def config2driver(config, keysys, ions, grid, pplist = None):
    #-----------------------------------------------------------------------
    # nr = config[keysys]["grid"]["nr"]
    # spacing = config[keysys]["grid"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
    # print('keysys', keysys, config[keysys])
    # print_conf(config[keysys])
    emb_ke_kwargs = config['GSYSTEM']["kedf"]
    emb_xc_kwargs = config['GSYSTEM']["exc"]
    gsystem_ecut = config['GSYSTEM']["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]
    ecut = config[keysys]["grid"]["ecut"] * ENERGY_CONV["eV"]["Hartree"]

    ke_kwargs = config[keysys]["kedf"]
    xc_kwargs = config[keysys]["exc"]
    cellcut = np.array(config[keysys]["cell"]["cut"]) * LEN_CONV["Angstrom"]["Bohr"]
    cellsplit= config[keysys]["cell"]["split"]
    index = config[keysys]["cell"]["index"]
    embed = config[keysys]["embed"]
    initial = config[keysys]["density"]["initial"]
    infile = config[keysys]["density"]["file"]
    use_gaussians = config[keysys]["density"]["use_gaussians"]
    gaussians_rcut = config[keysys]["density"]["gaussians_rcut"] * LEN_CONV["Angstrom"]["Bohr"]
    gaussians_sigma = config[keysys]["density"]["gaussians_sigma"]
    gaussians_scale = config[keysys]["density"]["gaussians_scale"]
    technique = config[keysys]["technique"]
    opt_options = config[keysys]["opt"]
    calculator = config[keysys]["calculator"]
    mix_kwargs = config[keysys]["mix"]
    basefile = config[keysys]["basefile"]
    prefix = config[keysys]["prefix"]
    kpoints = config[keysys]["kpoints"]
    exttype = config[keysys]["exttype"]
    pme = config["MATH"]["linearie"]
    atomicfiles = config[keysys]["density"]["atomic"]
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
    print('index', index)
    subsys = SubCell(ions, grid, index = index, cellcut = cellcut, cellsplit = cellsplit, optfft = True, gaussian_options = gaussian_options)
    if infile : # initial='Read'
        subsys.density[:] = io.read_density(infile)
    elif initial == 'Atomic' and len(atomicfiles) > 0 :
        atomicd = AtomicDensity(files = atomicfiles)
        subsys.density[:] = atomicd.guess_rho(subsys.ions, subsys.grid)
    elif initial or technique == 'OF' :
        atomicd = AtomicDensity()
        subsys.density[:] = atomicd.guess_rho(subsys.ions, subsys.grid)
        dftpy_opt(subsys.ions, subsys.density, pplist)
    #Embedding Functional---------------------------------------------------
    emb_funcdicts = {}
    if exttype is not None :
        embed = ['KE']
        if not exttype & 1 : embed.append('PSEUDO')
        if not exttype & 2 : embed.append('HARTREE')
        if not exttype & 4 : embed.append('XC')
    exttype = 7
    if 'KE' in embed :
        ke_emb = KEDF(**emb_ke_kwargs)
        emb_funcdicts['KE'] = ke_emb
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
    print('exttype', exttype, embed)
    emb_evaluator = Evaluator(**emb_funcdicts)
    #Subsystem Functional---------------------------------------------------
    ke_sub_kwargs = {'name' :'vW'}
    ke_sub = KEDF(**ke_sub_kwargs)
    sub_funcdicts = {'KE' :ke_sub}
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
    #-----------------------------------------------------------------------
    if mix_kwargs['scheme'] == 'Pulay' :
        mixer = PulayMixer(**mix_kwargs)
    elif mix_kwargs['scheme'] == 'Linear' :
        mixer = LinearMixer(**mix_kwargs)
    else :
        raise AttributeError("!!!ERROR : NOT support ", mix_kwargs['scheme'])

    energy_evaluator = EnergyEvaluatorMix(embed_evaluator = emb_evaluator, sub_evaluator = sub_evaluator, ke_evaluator = ke_evaluator)

    def get_dftpy_enginer():
        if opt_options['opt_method'] == 'full' :
            mixer = LinearMixer(predtype = None, coef = [1.0], predecut = None, delay = 1)
        opt_options['econv'] *= subsys.ions.nat
        enginer = DFTpyOF(options = opt_options, subcell = subsys, mixer = mixer)
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
            if abs(8 * ecut - gsystem_ecut) < 1.0 :
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

    driver = OptDriver(energy_evaluator = energy_evaluator, calculator = enginer)
    return driver


def dftpy_opt(ions, rho, pplist):
    from dftpy.interface import OptimizeDensityConf
    from dftpy.config import DefaultOption, OptionFormat
    from dftpy.system import System
    #-----------------------------------------------------------------------
    pseudo = LocalPP(grid = rho.grid, ions=ions, PP_list=pplist,PME=True)
    hartree = Hartree()
    xc_kwargs = {"x_str":'gga_x_pbe','c_str':'gga_c_pbe'}
    xc = XC(**xc_kwargs)
    ke_kwargs = {'name' :'GGA', 'k_str' : 'REVAPBEK'}
    # ke_kwargs = {'name' :'TFvW', 'y' :0.2}
    ke = KEDF(**ke_kwargs)
    funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
    evaluator = Evaluator(**funcdicts)
    #-----------------------------------------------------------------------
    conf = DefaultOption()
    conf['JOB']['calctype'] = 'Density'
    conf['OUTPUT']['time'] = False
    conf['OPT'] = {"method" :'CG-HS', "maxiter": 1000, "econv": 1.0e-6*ions.nat}
    conf = OptionFormat(conf)
    struct = System(ions, rho.grid, name='density', field=rho)
    opt = OptimizeDensityConf(conf, struct, evaluator)
    rho[:] = opt['density']
    return rho

def get_forces(opt_drivers = None, gsystem = None, linearii=True):
    forces = gsystem.get_forces(linearii = linearii)
    for i, driver in enumerate(opt_drivers):
        fs = driver.calculator.get_forces()
        ind = driver.calculator.subcell.ions_index
        # print('ind', ind)
        # print('fs', fs)
        forces[ind] += fs
    return forces

def get_stress():
    pass
