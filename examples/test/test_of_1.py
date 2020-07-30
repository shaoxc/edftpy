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
from edftpy.evaluator import Evaluator, EnergyEvaluatorMix, EvaluatorOF
from edftpy.enginer.driver import OptDriver
from edftpy.enginer.of_dftpy import DFTpyOF
from edftpy.density.init_density import AtomicDensity
from edftpy.subsystem.subcell import SubCell, GlobalCell
from edftpy.mixer import PulayMixer, LinearMixer

class Test(unittest.TestCase):
    def test_optim(self):
        data_path = os.environ.get('EDFTPY_DATA_PATH')
        if not data_path : data_path = 'DATA/'
        if not os.path.exists(data_path) : data_path = '../DATA/'
        data_path += '/'
        path_pp = data_path
        path_pos = data_path

        pp_al ='Al_OEPP_lda.recpot'
        posfile='fcc.vasp'

        ions = io.read(path_pos+posfile, names=['Al'])
        gsystem = GlobalCell(ions, grid = None, ecut = 22.05, full = False, optfft = True)
        grid = gsystem.grid
        ############################## Functionals  ##############################
        pplist = {'Al': path_pp+pp_al}
        pseudo = LocalPP(grid = grid, ions=ions,PP_list=pplist,PME=True)
        hartree = Hartree()
        xc_kwargs = {"x_str":'lda_x','c_str':'lda_c_pz'}
        xc = XC(**xc_kwargs)
        emb_ke_kwargs = {'name' :'TF'}
        ke = KEDF(**emb_ke_kwargs)
        funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
        total_evaluator = Evaluator(**funcdicts)
        atomicd = AtomicDensity()
        #-----------------------------------------------------------------------
        gsystem.total_evaluator = total_evaluator
        #-----------------------------------------------------------------------

        def test_ke(ke_kwargs, method):
            index_a = None
            subsys_a, driver_a = self.gen_sub_of(ions, grid, pplist, index_a, atomicd, xc_kwargs, ke_kwargs, emb_ke_kwargs = emb_ke_kwargs, gsystem = gsystem, method = method)
            opt_drivers = [driver_a]
            optimization_options = {'econv' : 1e-6, 'maxiter' : 70}
            optimization_options["econv"] *= ions.nat
            opt = Optimization(opt_drivers = opt_drivers, options = optimization_options)
            opt.optimize(gsystem = gsystem)
            rho = opt.density
            energy = opt.energy
            ewald_ = ewald(rho=rho, ions=ions, PME=True)
            energy += ewald_.energy
            return energy

        for method in ['full', 'part', 'hamiltonian'] :
            # Test TFvW-KE
            ke_kwargs = {'name' :'TF'}
            energy = test_ke(ke_kwargs, method)
            print('method = ', method, 'TFvW : energy', energy)
            self.assertTrue(np.isclose(energy, -8.281114354275829, rtol = 1E-3))
            # Test vW-KE
            ke_kwargs = None
            energy = test_ke(ke_kwargs, method)
            print('method = ', method, 'vW : energy', energy)
            self.assertTrue(np.isclose(energy, -11.394097752526489, rtol = 1E-3))

    def gen_sub_of(self, ions, grid, pplist = None, index = None, atomicd = None, xc_kwargs = {}, ke_kwargs = {}, emb_ke_kwargs = {}, gsystem = None, method = 'part', **kwargs):
        if atomicd is None :
            atomicd = AtomicDensity()
        #-----------------------------------------------------------------------
        if method == 'full' :
            if ke_kwargs is None or len(ke_kwargs) == 0 :
                ke_evaluator = None
            else :
                ke_evaluator = KEDF(**ke_kwargs)
            ke_sub_a = KEDF(name='vW')
            sub_funcdicts_a = {'KE' :ke_sub_a}
            sub_evaluator_a = Evaluator(**sub_funcdicts_a)

            ke_emb_a = KEDF(**emb_ke_kwargs)
            emb_funcdicts_a = {'KE' :ke_emb_a}
            emb_evaluator_a = Evaluator(**emb_funcdicts_a)

            subsys_a = SubCell(ions, grid, index = index, cellcut = [0.0, 0.0, 10.5], optfft = True)
            ions_a = subsys_a.ions
            rho_a = subsys_a.density
            rho_a[:] = atomicd.guess_rho(ions_a, subsys_a.grid)
            options = {"method" :'CG-HS', "maxiter": 220, "econv": 1.0e-6, "ncheck": 2, "opt_method" : 'full'}
            energy_evaluator = EnergyEvaluatorMix(embed_evaluator = emb_evaluator_a, ke_evaluator = ke_evaluator, **kwargs)
            evaluator_of = EvaluatorOF(grid = subsys_a.grid, sub_evaluator = sub_evaluator_a, gsystem = gsystem)
            # mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0], maxm = 7, coef = [0.8], predecut = 0, delay = 1)
            mixer = LinearMixer(predtype = None, coef = [1.0], predecut = None, delay = 1)
        else : # part, hamiltonian
            if ke_kwargs is None or len(ke_kwargs) == 0 :
                sub_evaluator_a = None
            else :
                ke_sub_a = KEDF(**ke_kwargs)
                sub_funcdicts_a = {'KE' :ke_sub_a}
                sub_evaluator_a = Evaluator(**sub_funcdicts_a)

            ke_emb_a = KEDF(**emb_ke_kwargs)
            emb_funcdicts_a = {'KE' :ke_emb_a}
            emb_evaluator_a = Evaluator(**emb_funcdicts_a)

            subsys_a = SubCell(ions, grid, index = index, cellcut = [0.0, 0.0, 10.5], optfft = True)
            ions_a = subsys_a.ions
            rho_a = subsys_a.density
            rho_a[:] = atomicd.guess_rho(ions_a, subsys_a.grid)
            options = {"method" :'CG-HS', "maxiter": 220, "econv": 1.0e-6, "ncheck": 2, "opt_method" : method}
            ke_evaluator = KEDF(name='vW')
            energy_evaluator = EnergyEvaluatorMix(embed_evaluator = emb_evaluator_a, **kwargs)
            evaluator_of = EvaluatorOF(grid = subsys_a.grid, sub_evaluator = sub_evaluator_a, ke_evaluator = ke_evaluator, gsystem = gsystem)
            mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0], maxm = 7, coef = [0.8], predecut = 0, delay = 1)
        of_enginer_a = DFTpyOF(mixer = mixer, options = options, subcell = subsys_a, evaluator_of = evaluator_of)
        driver_a = OptDriver(energy_evaluator = energy_evaluator, calculator = of_enginer_a)
        return subsys_a, driver_a


if __name__ == "__main__":
    unittest.main()
