import os
import numpy as np
import unittest
from functools import partial

from dftpy.optimization import Optimization
from dftpy.ions import Ions
from ase.build import bulk

from edftpy.functional import LocalPP, KEDF, Hartree, XC, Ewald
from edftpy.evaluator import Evaluator
from edftpy.density import DensityGenerator
from edftpy.subsystem.subcell import GlobalCell

data_path = os.environ.get('EDFTPY_DATA_PATH')
if not data_path : data_path = 'DATA/'
if not os.path.exists(data_path) : data_path = '../DATA/'
pp_al = data_path + '/Al_OEPP_lda.recpot'
ions = Ions.from_ase(bulk('Al', 'fcc', a=4.05, cubic=True))

class Test(unittest.TestCase):
    def test_optim(self):
        gsystem = GlobalCell(ions, grid = None, spacing = 0.3, full = False, optfft = True)
        grid = gsystem.grid
        ############################## Functionals  ##############################
        pp_list = {'Al': pp_al}
        pseudo = LocalPP(grid = grid, ions=ions, PP_list=pp_list, PME=True)
        ke = KEDF(name='WT')
        xc_kwargs = {"libxc":['lda_x', 'lda_c_pz']}
        xc = XC(**xc_kwargs)
        hartree = Hartree()
        ewald = Ewald(ions=ions, grid = grid, PME=True)
        funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo, 'EWALD' : ewald}
        #-----------------------------------------------------------------------
        atomicd = DensityGenerator()
        rho_ini = atomicd.guess_rho(ions, grid)
        #-----------------------------------------------------------------------
        evaluator_of = Evaluator(**funcdicts)
        evaluator = partial(evaluator_of.compute, gather = True)
        optimization_options = {'econv' : 1e-6, 'maxfun' : 50, 'maxiter' : 100}
        optimization_options["econv"] *= ions.nat
        opt = Optimization(EnergyEvaluator= evaluator, optimization_options = optimization_options, optimization_method = 'CG-HS')
        opt.optimize_rho(guess_rho=rho_ini)
        energy = opt.functional.energy
        self.assertTrue(np.isclose(energy, -8.343674099220022))


if __name__ == "__main__":
    unittest.main()
