import os
import copy
import numpy as np
import unittest

from dftpy.constants import ENERGY_CONV
from dftpy.formats import io
from dftpy.ewald import ewald
from dftpy.optimization import Optimization

from edftpy.pseudopotential import LocalPP
from edftpy.kedf import KEDF
from edftpy.hartree import Hartree
from edftpy.xc import XC
from edftpy.evaluator import Evaluator
from edftpy.density.init_density import AtomicDensity
from edftpy.subsystem.subcell import GlobalCell

class Test(unittest.TestCase):
    def test_optim(self):
        data_path = os.environ.get('EDFTPY_DATA_PATH')
        if not data_path : data_path = 'DATA/'
        path_pp = data_path
        path_pos = data_path

        pp_al ='Al_lda.oe01.recpot'
        posfile='fcc.vasp'

        ions = io.read(path_pos+posfile, names=['Al'])

        gsystem = GlobalCell(ions, grid = None, spacing = 0.3, full = False, optfft = True)
        grid = gsystem.grid
        ############################## Functionals  ##############################
        pp_list = {'Al': path_pp+pp_al}
        pseudo = LocalPP(grid = grid, ions=ions, PP_list=pp_list, PME=True)
        ke = KEDF(name='WT')
        xc_kwargs = {"x_str":'lda_x','c_str':'lda_c_pz'}
        xc = XC(**xc_kwargs)
        hartree = Hartree()
        funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'pseudo' :pseudo}
        #-----------------------------------------------------------------------
        atomicd = AtomicDensity()
        rho_ini = atomicd.guess_rho(ions, grid)
        #-----------------------------------------------------------------------
        evaluator = Evaluator(**funcdicts)
        optimization_options = {'econv' : 1e-6, 'maxfun' : 50, 'maxiter' : 100}
        optimization_options["econv"] *= ions.nat
        opt = Optimization(EnergyEvaluator= evaluator, optimization_options = optimization_options, optimization_method = 'CG-HS')
        new_rho = opt.optimize_rho(guess_rho=rho_ini)
        ewald_ = ewald(rho=new_rho, ions=ions, PME=True)
        energy = opt.functional.energy + ewald_.energy
        self.assertTrue(np.isclose(energy, -8.343671094488517))


if __name__ == "__main__":
    unittest.main()
