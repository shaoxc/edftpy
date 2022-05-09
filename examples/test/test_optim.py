import os
import numpy as np
import unittest
from functools import partial

from dftpy.formats import io
from dftpy.optimization import Optimization

from edftpy.functional import LocalPP, KEDF, Hartree, XC, Ewald
from edftpy.evaluator import Evaluator
from edftpy.density import AtomicDensity
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
        # xc_kwargs = {"x_str":'lda_x','c_str':'lda_c_pz'}
        xc_kwargs = {"libxc":['lda_x', 'lda_c_pz']}
        xc = XC(**xc_kwargs)
        hartree = Hartree()
        ewald = Ewald(ions=ions, grid = grid, PME=True)
        funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo, 'EWALD' : ewald}
        #-----------------------------------------------------------------------
        atomicd = AtomicDensity()
        rho_ini = atomicd.guess_rho(ions, grid)
        #-----------------------------------------------------------------------
        evaluator_of = Evaluator(**funcdicts)
        evaluator = partial(evaluator_of.compute, gather = True)
        optimization_options = {'econv' : 1e-6, 'maxfun' : 50, 'maxiter' : 100}
        optimization_options["econv"] *= ions.nat
        opt = Optimization(EnergyEvaluator= evaluator, optimization_options = optimization_options, optimization_method = 'CG-HS')
        opt.optimize_rho(guess_rho=rho_ini)
        energy = opt.functional.energy
        self.assertTrue(np.isclose(energy, -8.343671094488517))


if __name__ == "__main__":
    unittest.main()
