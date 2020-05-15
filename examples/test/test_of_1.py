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
from edftpy.evaluator import Evaluator
from edftpy.enginer.driver import OptDriver
from edftpy.enginer.of_dftpy import DFTpyOF
from edftpy.density.init_density import AtomicDensity
from edftpy.subsystem.subcell import SubCell, GlobalCell

class Test(unittest.TestCase):
    def test_optim(self):
        data_path = os.environ.get('EDFTPY_DATA_PATH')
        if not data_path : data_path = 'DATA/'
        data_path += '/'
        path_pp = data_path
        path_pos = data_path

        pp_al ='Al_lda.oe01.recpot'
        # posfile='fcc.vasp'
        posfile='sub_al_al.vasp'

        ions = io.read(path_pos+posfile, names=['Al'])

        # gsystem = GlobalCell(ions, grid = None, spacing = 0.4, full = False, optfft = True)
        gsystem = GlobalCell(ions, grid = None, ecut = 22.05, full = False, optfft = True)
        grid = gsystem.grid
        ############################## Functionals  ##############################
        PP_list = {'Al': path_pp+pp_al}
        PSEUDO = LocalPP(grid = grid, ions=ions,PP_list=PP_list,PME=True)

        ke = KEDF(name='TF')
        ke2 = KEDF(name='TF')
        ke3 = KEDF(name='TF')
        xc_kwargs = {"x_str":'lda_x','c_str':'lda_c_pz'}
        xc = XC(**xc_kwargs)
        xc2 = XC(**xc_kwargs)
        xc3 = XC(**xc_kwargs)
        hartree = Hartree()
        #-----------------------------------------------------------------------
        opt_drivers = []
        funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :PSEUDO}
        embed_funcdicts = {'KE' :ke2, 'XC' :xc2}
        sub_funcdicts = {'KE' :ke3, 'XC' :xc3}

        total_evaluator = Evaluator(**funcdicts)
        embed_evaluator = Evaluator(**embed_funcdicts)
        sub_evaluator = Evaluator(**sub_funcdicts)

        options = {"method" :'CG-HS', "maxiter": 60, "econv": 1.0e-5, "ncheck": 2, 'opt_method' :'full'}
        of_enginer = DFTpyOF(options = options)
        driver = OptDriver(embed_evaluator = embed_evaluator, sub_evaluator = sub_evaluator, calculator = of_enginer)
        opt_drivers = [driver]
        #-----------------------------------------------------------------------
        subsys_a = SubCell(ions, grid, index = None, cellcut = [0.0, 0.0, 10.5], optfft = True)
        ions_a = subsys_a.ions
        rho_a = subsys_a.density
        # denfiles= {'Al': data_path + 'Al.LDA.gz'} 
        # ftypes = {'Al': 'xml'}
        # atomicd = AtomicDensity(denfiles, ftypes = ftypes)
        atomicd = AtomicDensity()
        rho_a[:] = atomicd.guess_rho(ions_a, subsys_a.grid)
        rho_ini = [rho_a]
        #-----------------------------------------------------------------------
        gsystem.total_evaluator = total_evaluator
        #-----------------------------------------------------------------------
        # optimization_options = {'econv' : 1e-8, 'maxiter' : 10}
        optimization_options = {'econv' : 1e-4, 'maxiter' : 10}
        optimization_options["econv"] *= ions.nat
        opt = Optimization(opt_drivers = opt_drivers, options = optimization_options)
        opt.optimize(gsystem = gsystem, guess_rho=rho_ini)
        rho = opt.density
        energy = opt.energy
        ewald_ = ewald(rho=rho, ions=ions, PME=True)
        energy += ewald_.energy
        print(energy)
        self.assertTrue(np.isclose(energy, -9.220, rtol = 1E-3))


if __name__ == "__main__":
    unittest.main()
