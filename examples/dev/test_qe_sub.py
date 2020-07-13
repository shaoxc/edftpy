import numpy as np
import copy
import os
import unittest

from dftpy.constants import ENERGY_CONV, LEN_CONV
from dftpy.formats import io
from dftpy.formats import ase_io
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
from edftpy.enginer.ks_pwscf import PwscfKS
from edftpy.mixer import LinearMixer, PulayMixer
from edftpy.interface import get_forces
import ase.io.espresso as ase_io_driver

class Test(unittest.TestCase):
    def test_optim(self):
        data_path = os.environ.get('EDFTPY_DATA_PATH')
        if not data_path : data_path = 'DATA/'
        if not os.path.exists(data_path) : data_path = '../DATA/'
        data_path += '/'
        path_pp = data_path

        pp_al = 'al_gga_blps.upf'
        pp_c = 'C_ONCV_PBE-1.2.upf'
        pp_o = 'O_ONCV_PBE-1.2.upf'
        # posfile= data_path + 'fcc.vasp'
        # posfile= data_path + 'al32.vasp'
        # posfile= 'castep_in.cell'
        posfile= 'qe_in.in'
        # posfile= data_path + 'co2.vasp'
        #-----------------------------------------------------------------------
        pplist = {'Al': path_pp+pp_al, 'C' :path_pp + pp_c, 'O': path_pp + pp_o}
        # pplist = {'Al': path_pp+pp_al}
        atomicd = AtomicDensity()

        # ions = io.read(posfile)
        ions = ase_io.ase_read(posfile)
        gsystem = GlobalCell(ions, grid = None, ecut = 80.0, full = False, optfft = True)
        # gsystem = GlobalCell(ions, grid = None, ecut = 22.0, full = False, optfft = True)
        grid = gsystem.grid
        pseudo = LocalPP(grid = grid, ions=ions,PP_list=pplist,PME=True)
        hartree = Hartree()
        xc_kwargs = {"x_str":'gga_x_pbe','c_str':'gga_c_pbe'}
        # xc_kwargs = {"x_str":'gga_x_pbe_r','c_str':'gga_c_pbe'}
        xc = XC(**xc_kwargs)
        emb_ke_kwargs = {'name' :'GGA', 'k_str' : 'REVAPBEK'}
        ke = KEDF(**emb_ke_kwargs)
        funcdicts = {'KE' :ke, 'XC' :xc, 'HARTREE' :hartree, 'PSEUDO' :pseudo}
        total_evaluator = Evaluator(**funcdicts)
        gsystem.total_evaluator = total_evaluator
        ############################## Subsytem ##############################
        ke_kwargs = {}
        index_a = None
        subsys_a, driver_a = self.gen_sub_ks(ions, grid, pplist, index_a, atomicd, xc_kwargs, ke_kwargs, emb_ke_kwargs = emb_ke_kwargs)
        opt_drivers = [driver_a]
        optimization_options = {'econv' : 1e-6, 'maxiter' : 80}
        # optimization_options = {'econv' : 1e-6, 'maxiter' : 2}
        optimization_options["econv"] *= ions.nat
        opt = Optimization(opt_drivers = opt_drivers, options = optimization_options, gsystem = gsystem)
        opt.optimize()
        rho = opt.density
        energy = opt.energy
        print('eee', energy, total_evaluator(rho).energy)
        ewald_ = ewald(rho=rho, ions=ions, PME=True)
        print('Final energy (a.u.)', energy + ewald_.energy)
        print('Final energy (eV)', (energy + ewald_.energy) * ENERGY_CONV['Hartree']['eV'])
        forces = driver_a.calculator.get_forces(icalc = 0)
        print('forces', forces)
        print('forces', forces*27.21138/0.529177)
        forces = get_forces(opt_drivers, gsystem, linearii = True)
        print('forces1', forces)
        print('forces1', forces*27.21138/0.529177)

    def gen_sub_ks(self, ions, grid, pplist = None, index = None, atomicd = None, xc_kwargs = {}, ke_kwargs = {}, emb_ke_kwargs = {}, gaussian_options= None, **kwargs):
        ke_emb_a = KEDF(**emb_ke_kwargs)
        xc_emb_a = XC(**xc_kwargs)
        # hartree = Hartree()
        # emb_funcdicts_a = {'KE' :ke_emb_a, 'XC' :xc_emb_a, 'HARTREE' :hartree}
        emb_funcdicts_a = {'KE' :ke_emb_a, 'XC' :xc_emb_a}
        # emb_funcdicts_a = {'KE' :ke_emb_a}
        emb_evaluator_a = Evaluator(**emb_funcdicts_a)
        sub_evaluator_a = None

        subsys_a = SubCell(ions, grid, index = index, cellcut = [0.0, 0.0, 0.0], optfft = True, gaussian_options = gaussian_options)
        rho_a = subsys_a.density
        energy_evaluator = EnergyEvaluatorMix(embed_evaluator = emb_evaluator_a, sub_evaluator = sub_evaluator_a, **kwargs)
        kgrid = np.maximum(np.round(40/grid.latparas), 1).astype(np.int)
        kgrid = list(kgrid)
        kgrid = [1, 1, 1]
        print(kgrid, 'kgrid')
        ecut = 100
        cell_params = {
                'pseudopotentials' : pplist, 
                'kpts' : kgrid, 
                'koffset': (0, 0, 0),
                }

        params = {
            'system' : {
                'ecutwfc' : ecut,
                # 'nr1' : subsys_a.grid.nr[0],
                # 'nr2' : subsys_a.grid.nr[1],
                # 'nr3' : subsys_a.grid.nr[2], 
                }
            }
        # params = {}
        # mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, 0.6, 1.0], maxm = 7, coef = [0.7], predecut = None, delay = 1, restarted = False)
        mixer = PulayMixer(predtype = 'inverse_kerker', predcoef = [0.2, 0.8], maxm = 7, coef = [0.7], predecut = None, delay = 1, restarted = False)
        ks_enginer_a = PwscfKS(prefix = 'qe_sub.in', subcell = subsys_a, cell_params = cell_params, params = params, exttype = 3,
                base_in_file = 'qe_in.in', mixer = mixer)
        opt_options = {}
        driver_a = OptDriver(energy_evaluator = energy_evaluator, calculator = ks_enginer_a, options = opt_options)
        rho0 = np.mean(rho_a)
        kf = (3.0 * rho0 * np.pi ** 2) ** (1.0 / 3.0)
        print('kf', kf)
        ks_enginer_a.mixer.predcoef=[1.0,kf,1.0]
        return subsys_a, driver_a


if __name__ == "__main__":
    unittest.main()
