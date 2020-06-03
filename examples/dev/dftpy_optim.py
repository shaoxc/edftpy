import numpy as np
from dftpy.optimization import Optimization
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.constants import ENERGY_CONV
from dftpy.formats import io
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import bestFFTsize
from dftpy.time_data import TimeData
from dftpy.pseudo import LocalPseudo

def test_optim():
    path_pp='../DATA/'
    path_pos='../DATA/'
    pp_al ='Al_OEPP_lda.recpot'
    posfile='fcc.vasp'
    ions = io.read(path_pos+posfile)
    lattice = ions.pos.cell.lattice
    metric = np.dot(lattice.T, lattice)
    gap = 0.4
    nr = np.zeros(3, dtype = 'int32')
    for i in range(3):
        nr[i] = int(np.sqrt(metric[i, i])/gap)
    print('The initial grid size is ', nr)
    for i in range(3):
        nr[i] = bestFFTsize(nr[i])
    print('The final grid size is ', nr)
    grid = DirectGrid(lattice=lattice, nr=nr, units=None, full=False)
    zerosA = np.zeros(grid.nnr, dtype=float)
    rho_ini = DirectField(grid=grid, griddata_F=zerosA, rank=1)

    PP_list = {'Al': path_pp+pp_al}

    PSEUDO = LocalPseudo(grid = grid, ions=ions,PP_list=PP_list,PME=True)
    optional_kwargs = {}
    optional_kwargs = {'x' :1.0, 'y' :1.0, 'sigma' :0.025, 'interp' :'hermite', 'kdd' :3, 'nsp':60, 'fd':3, 'kfmin' :1E-3, 'kfmax' :2.5}
    # KE = FunctionalClass(type='KEDF',name='x_TF_y_vW',optional_kwargs=optional_kwargs)
    # KE = FunctionalClass(type='KEDF',name='LMGPA',optional_kwargs=optional_kwargs)
    KE = FunctionalClass(type='KEDF',name='TF',optional_kwargs=optional_kwargs)
    optional_kwargs = {"x_str":'lda_x','c_str':'lda_c_pz'}
    XC = FunctionalClass(type='XC',name='LDA')
    HARTREE = FunctionalClass(type='HARTREE')

    charge_total = 0.0
    for i in range(ions.nat) :
        charge_total += ions.Zval[ions.labels[i]]
    rho_ini[:] = charge_total/ions.pos.cell.volume
    #-----------------------------------------------------------------------
    funcDict = {'KE' :KE, 'XC' :XC, 'HARTREE' :HARTREE, 'PSEUDO' :PSEUDO}
    E_v_Evaluator = TotalEnergyAndPotential(**funcDict)
    optimization_options = {'econv' : 1e-5, 'maxfun' : 50, 'maxiter' : 100}
    optimization_options["econv"] *= ions.nat
    opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_options = optimization_options, 
            optimization_method = 'CG-HS')
            # optimization_method = 'TN')
    new_rho = opt.optimize_rho(guess_rho=rho_ini)
    print('Calc Energy')
    Enew = E_v_Evaluator.Energy(rho=new_rho, ions=ions, usePME = True)
    print('Energy New (a.u.)', Enew)
    print('Energy New (eV)', Enew * ENERGY_CONV['Hartree']['eV'])
    print('Energy New (eV/atom)', Enew * ENERGY_CONV['Hartree']['eV']/ions.nat)
    print('-' * 31, 'Time information', '-' * 31)
    print("{:28s}{:24s}{:20s}".format('Label', 'Cost(s)', 'Number'))
    for key in TimeData.cost :
        print("{:28s}{:<24.4f}{:<20d}".format(key, TimeData.cost[key], TimeData.number[key]))
    print('-' * 80)


if __name__ == "__main__":
    test_optim()
