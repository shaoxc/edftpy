import numpy as np
import copy
import os
import unittest

from dftpy.constants import ENERGY_CONV
from dftpy.formats import io
from dftpy.ewald import ewald

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
from edftpy.enginer.ks_castep import CastepKS
from edftpy.mixer import LinearMixer, PulayMixer

class Test(unittest.TestCase):
    def test_optim(self):
        data_path = os.environ.get('EDFTPY_DATA_PATH')
        if not data_path : data_path = 'DATA/'
        if not os.path.exists(data_path) : data_path = '../DATA/'
        data_path += '/'
        path_pp = data_path
        path_pos = data_path

        pp_al ='al.gga.recpot'
        posfile='al_111.vasp'
        posfile='fcc.vasp'

        pplist = {'Al': path_pp+pp_al}
        denfiles= {'Al': data_path + 'al.gga.recpot.list'}
        ftypes = {'Al': 'list'}
        atomicd = AtomicDensity(denfiles, ftypes = ftypes, direct = False)
        # atomicd = AtomicDensity()

        # gaussian_options = { 'Al' : {'rcut' : 6.0, 'sigma' : 0.3, 'scale' : 10} }
        gaussian_options = None

        ions = io.read(path_pos+posfile)
        gsystem = GlobalCell(ions, grid = None, ecut = 22.0, full = False, optfft = True)
        grid = gsystem.grid
        ############################## Functionals  ##############################
        pseudo = LocalPP(grid = grid, ions=ions,PP_list=pplist,PME=True)
        hartree = Hartree()
        # xc_kwargs = {"x_str":'lda_x','c_str':'lda_c_pz'}
        xc_kwargs = {"x_str":'gga_x_pbe','c_str':'gga_c_pbe'}
        xc = XC(**xc_kwargs)
        emb_ke_kwargs = {'name' :'GGA', 'k_str' : 'REVAPBEK'}
        ke = KEDF(**emb_ke_kwargs)
        funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
        total_evaluator = Evaluator(**funcdicts)
        ############################## Subsytem ##############################
        # ke_kwargs = {'name' :'LMGPA', 'y' :0.0, 'interp' :'hermite', 'kdd' :3, 'ratio':1.15, 'fd':3, 'kfmin' :1E-3, 'kfmax' :4.0, 'lumpfactor' : None, 'ldw' : None}
        ke_kwargs = {'name' :'TF', 'x' :1.0}
        index_a = None
        subsys_a, driver_a = self.gen_sub_of(ions, grid, pplist, index_a, atomicd, xc_kwargs, ke_kwargs, emb_ke_kwargs = emb_ke_kwargs, gaussian_options = gaussian_options)
        # self.dftpy_opt(subsys_a.ions, subsys_a.density, pplist)
        # subsys_a.density[:] = driver_a.filter_density(subsys_a.density)
        #-----------------------------------------------------------------------
        opt_drivers = [driver_a]
        rho_ini = [subsys_a.density]
        #-----------------------------------------------------------------------
        gsystem.total_evaluator = total_evaluator
        #-----------------------------------------------------------------------
        optimization_options = {'econv' : 1e-8, 'maxiter' : 80}
        optimization_options["econv"] *= ions.nat
        opt = Optimization(opt_drivers = opt_drivers, options = optimization_options)
        opt.optimize(gsystem = gsystem, guess_rho=rho_ini)
        rho = opt.density
        energy = opt.energy
        print('eee', energy, total_evaluator(rho).energy)
        ewald_ = ewald(rho=rho, ions=ions, PME=True)
        energy += ewald_.energy
        print(energy)
        io.write('final_al.xsf', rho, ions)

    def gen_sub_of(self, ions, grid, pplist = None, index = None, atomicd = None, xc_kwargs = {}, ke_kwargs = {}, emb_ke_kwargs = {}, gaussian_options= None, **kwargs):
        if atomicd is None :
            atomicd = AtomicDensity()
        #-----------------------------------------------------------------------
        ke_sub_a = KEDF(**ke_kwargs)
        # xc_sub_a = XC(**xc_kwargs)
        sub_funcdicts_a = {'KE' :ke_sub_a}
        sub_evaluator_a = Evaluator(**sub_funcdicts_a)

        ke_emb_a = KEDF(**emb_ke_kwargs)
        # xc_emb_a = XC(**xc_kwargs)
        emb_funcdicts_a = {'KE' :ke_emb_a}
        emb_evaluator_a = Evaluator(**emb_funcdicts_a)
        # sub_evaluator_a = None

        subsys_a = SubCell(ions, grid, index = index, cellcut = [0.0, 0.0, 10.5], optfft = True, gaussian_options = gaussian_options)
        ions_a = subsys_a.ions
        rho_a = subsys_a.density
        rho_a[:] = atomicd.guess_rho(ions_a, subsys_a.grid)
        options = {"method" :'CG-HS', "maxiter": 820, "econv": 1.0e-6*ions.nat, "ncheck": 2, "opt_method" : 'part'}
        # options = {"method" :'CG-HS', "maxiter": 820, "econv": 1.0e-8*ions.nat, "ncheck": 2, "opt_method" : 'hamiltonian'}
        ke_evaluator = KEDF(name='vW')
        # vw_kwargs = {'name' :'GGA', 'k_str' : 'STV', 'params' : [0.0, 1.0, 0.01]}
        # ke_evaluator = KEDF(**vw_kwargs)
        energy_evaluator = EnergyEvaluatorMix(embed_evaluator = emb_evaluator_a, sub_evaluator = sub_evaluator_a, ke_evaluator = ke_evaluator, **kwargs)
        opt_options = {}
        #mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.2], maxm = 7, coef = [0.2], predecut = 0, delay = 1)
        mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, 0.4], maxm = 7, coef = [0.2], predecut = None, delay = 0)
        # mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, 1.0], maxm = 7, coef = [0.4], predecut = None, delay = 0, restarted = True)
        of_enginer_a = DFTpyOF(options = options, ions = ions_a, gaussian_density = subsys_a.gaussian_density, mixer = mixer)
        driver_a = OptDriver(energy_evaluator = energy_evaluator, calculator = of_enginer_a, options = opt_options, mixer=mixer)
        return subsys_a, driver_a

    def dftpy_opt(self, ions, rho, pplist):
        from dftpy.interface import OptimizeDensityConf
        from dftpy.config import DefaultOption, OptionFormat
        from dftpy.system import System
        #-----------------------------------------------------------------------
        pseudo = LocalPP(grid = rho.grid, ions=ions, PP_list=pplist,PME=True)
        hartree = Hartree()
        xc_kwargs = {"x_str":'gga_x_pbe','c_str':'gga_c_pbe'}
        xc = XC(**xc_kwargs)
        # ke_kwargs = {'name' :'TF'}
        ke_kwargs = {'name' :'GGA', 'k_str' : 'REVAPBEK'}
        ke = KEDF(**ke_kwargs)
        funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
        evaluator = Evaluator(**funcdicts)
        #-----------------------------------------------------------------------
        conf = DefaultOption()
        conf['JOB']['calctype'] = 'Density'
        conf['OUTPUT']['time'] = False
        conf['OPT'] = {"method" :'CG-HS', "maxiter": 200, "econv": 1.0e-6*ions.nat}
        conf = OptionFormat(conf)
        struct = System(ions, rho.grid, name='density', field=rho)
        opt = OptimizeDensityConf(conf, struct, evaluator)
        rho[:] = opt['density']
        return rho


if __name__ == "__main__":
    # unittest.main()
    a = Test()
    a.test_optim()
    
