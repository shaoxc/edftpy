import numpy as np
import copy
import os
import unittest

from dftpy.constants import ENERGY_CONV
from dftpy.formats import io

from edftpy.utils.common import Field, Grid, Atoms
from edftpy.functional import LocalPP, KEDF, Hartree, XC
from edftpy.optimizer import Optimization
from edftpy.evaluator import EmbedEvaluator, EvaluatorOF, TotalEvaluator
from edftpy.density.init_density import AtomicDensity
from edftpy.subsystem.subcell import SubCell, GlobalCell
from edftpy.mixer import PulayMixer, LinearMixer
from edftpy.mpi import GraphTopo, MP

class Test(unittest.TestCase):
    def test_optim(self):
        xc_kwargs = {"x_str":'lda_x','c_str':'lda_c_pz', 'libxc' :False}
        for method in ['full', 'part', 'hamiltonian'] :
            # Test TFvW-KE
            ke_kwargs = {'name' :'TF'}
            opt = self.get_optimizer(ke_kwargs, xc_kwargs = xc_kwargs, method = method)
            energy = self.get_energy(opt)
            print('method = ', method, 'TFvW : energy', energy)
            self.assertTrue(np.isclose(energy, -8.281114354275829, rtol = 1E-3))
            # Test vW-KE
            ke_kwargs = None
            opt = self.get_optimizer(ke_kwargs, xc_kwargs = xc_kwargs, method = method)
            energy = self.get_energy(opt)
            print('method = ', method, 'vW : energy', energy)
            self.assertTrue(np.isclose(energy, -11.394097752526489, rtol = 1E-3))

    def get_optimizer(self, ke_kwargs, xc_kwargs = {}, method = 'full'):
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
        xc = XC(**xc_kwargs)
        emb_ke_kwargs = {'name' :'TF'}
        ke = KEDF(**emb_ke_kwargs)
        funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
        total_evaluator = TotalEvaluator(**funcdicts)
        #-----------------------------------------------------------------------
        gsystem.total_evaluator = total_evaluator
        graphtopo = GraphTopo()
        # nprocs = [0]
        # graphtopo.distribute_procs(nprocs)
        mp = MP(comm = graphtopo.comm_sub)
        gsystem.graphtopo = graphtopo
        #-----------------------------------------------------------------------
        index_a = None
        atomicd = AtomicDensity()
        driver_a = self.gen_sub_of(ions, grid, pplist, index_a, atomicd, xc_kwargs, ke_kwargs, emb_ke_kwargs = emb_ke_kwargs, gsystem = gsystem, method = method, mp = mp)
        drivers = [driver_a]
        graphtopo.build_region(grid=gsystem.grid, drivers=drivers)
        optimization_options = {'econv' : 1e-6, 'maxiter' : 70}
        optimization_options["econv"] *= gsystem.ions.nat
        opt = Optimization(gsystem = gsystem, drivers = drivers, options = optimization_options)
        return opt

    def get_energy(self, opt):
        opt.optimize()
        energy = opt.energy
        opt.stop_run()
        return energy

    def gen_sub_of(self, ions, grid, pplist = None, index = None, atomicd = None, xc_kwargs = {}, ke_kwargs = {}, emb_ke_kwargs = {}, gsystem = None, method = 'part', mp = None, **kwargs):
        if atomicd is None :
            atomicd = AtomicDensity()
        mixer = PulayMixer(predtype = 'kerker', predcoef = [0.8, 1.0], maxm = 7, coef = 0.8, predecut = 0, delay = 1)
        #-----------------------------------------------------------------------
        if ke_kwargs is None or len(ke_kwargs) == 0 :
            ke_evaluator = None
        else :
            ke_evaluator = KEDF(**ke_kwargs)

        ke_emb_a = KEDF(**emb_ke_kwargs)
        emb_funcdicts = {'KE' :ke_emb_a}

        ke_sub_kwargs = {'name' :'vW'}
        ke_sub = KEDF(**ke_sub_kwargs)

        sub_funcdicts = {}
        if method == 'full' :
            sub_funcdicts['KE'] = ke_sub
            evaluator_of = EvaluatorOF(gsystem = gsystem, **sub_funcdicts)
            embed_evaluator = EmbedEvaluator(ke_evaluator = ke_evaluator, **emb_funcdicts)
        else :
            sub_funcdicts['KE'] = ke_evaluator
            evaluator_of = EvaluatorOF(gsystem = gsystem, ke_evaluator = ke_sub, **sub_funcdicts)
            embed_evaluator = EmbedEvaluator(**emb_funcdicts)
        #-----------------------------------------------------------------------
        subsys_a = SubCell(ions, grid, index = index, cellcut = [0.0, 0.0, 10.5], optfft = True, mp = mp)
        ions_a = subsys_a.ions
        rho_a = subsys_a.density
        rho_a[:] = atomicd.guess_rho(ions_a, subsys_a.grid)
        options = {"method" :'CG-HS', "maxiter": 220, "econv": 1.0e-6, "ncheck": 2, "opt_method" : method}
        from edftpy.engine.driver import DriverOF
        of_enginer_a = DriverOF(evaluator = embed_evaluator, mixer = mixer, options = options, subcell = subsys_a, evaluator_of = evaluator_of)
        return of_enginer_a

    def tearDown(self):
        if os.path.isfile('sub_of.out'):
            os.remove('sub_of.out')


if __name__ == "__main__":
    unittest.main()
