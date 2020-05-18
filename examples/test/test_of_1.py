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
from edftpy.mixer import PulayMixer

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
        posfile='sub_al_al.vasp'

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
        ############################## Subsytem ##############################
        denfiles= {'Al': data_path + 'Al.LDA.gz'} 
        ftypes = {'Al': 'xml'}
        atomicd = AtomicDensity(denfiles, ftypes = ftypes)
        ke_kwargs = {}
        index_a = None
        subsys_a, driver_a = self.gen_sub_of(ions, grid, pplist, index_a, atomicd, xc_kwargs, ke_kwargs, emb_ke_kwargs = emb_ke_kwargs)
        opt_drivers = [driver_a]
        rho_ini = [subsys_a.density]
        #-----------------------------------------------------------------------
        gsystem.total_evaluator = total_evaluator
        #-----------------------------------------------------------------------
        optimization_options = {'econv' : 1e-8, 'maxiter' : 80}
        # optimization_options = {'econv' : 1e-8, 'maxiter' : 1}
        optimization_options["econv"] *= ions.nat
        opt = Optimization(opt_drivers = opt_drivers, options = optimization_options)
        opt.optimize(gsystem = gsystem, guess_rho=rho_ini)
        rho = opt.density
        energy = opt.energy
        ewald_ = ewald(rho=rho, ions=ions, PME=True)
        energy += ewald_.energy
        print(energy)
        self.assertTrue(np.isclose(energy, -10.117997349843236, rtol = 1E-3))

    def gen_sub_of(self, ions, grid, pplist = None, index = None, atomicd = None, xc_kwargs = {}, ke_kwargs = {}, emb_ke_kwargs = {}, **kwargs):
        if atomicd is None :
            atomicd = AtomicDensity()
        #-----------------------------------------------------------------------
        # ke_sub_a = KEDF(**ke_kwargs)
        # sub_funcdicts_a = {'KE' :ke_sub_a}
        # sub_evaluator_a = Evaluator(**sub_funcdicts_a)
        sub_evaluator_a = None

        ke_emb_a = KEDF(**emb_ke_kwargs)
        emb_funcdicts_a = {'KE' :ke_emb_a}
        emb_evaluator_a = Evaluator(**emb_funcdicts_a)

        # subsys_a = SubCell(ions, grid, index = index, cellcut = [0.0, 0.0, 10.5], optfft = True)
        subsys_a = SubCell(ions, grid, index = index, cellcut = [0.0, 0.0, 0.0], optfft = True)
        ions_a = subsys_a.ions
        rho_a = subsys_a.density
        rho_a[:] = atomicd.guess_rho(ions_a, subsys_a.grid)
        options = {"method" :'CG-HS', "maxiter": 220, "econv": 1.0e-6, "ncheck": 2}
        ke_evaluator = KEDF(name='vW')
        energy_evaluator = EnergyEvaluatorMix(embed_evaluator = emb_evaluator_a, sub_evaluator = sub_evaluator_a, ke_evaluator = ke_evaluator, **kwargs)
        mixer = PulayMixer(predtype = 'kerker', predcoef = [0.2, 1.0], maxm = 7, coef = [0.4], predecut = None, delay = 1)
        of_enginer_a = DFTpyOF(mixer = mixer, options = options)
        driver_a = OptDriver(energy_evaluator = energy_evaluator, calculator = of_enginer_a)
        return subsys_a, driver_a


if __name__ == "__main__":
    unittest.main()
