import numpy as np
import pwscfpy
from dftpy.constants import ENERGY_CONV, LEN_CONV
from dftpy.formats import ase_io

fname = 'qe_in.in'
pwscfpy.pwpy_pwscf(fname)

nr = np.zeros(3, dtype = 'int32')
pwscfpy.pwpy_mod.pwpy_get_grid(nr)

rho = np.zeros((np.prod(nr), 1), order = 'F')

pwscfpy.pwpy_mod.pwpy_get_rho(rho)

printout = 2; exxen = 0.0; extpot = np.zeros(np.prod(nr)); extene = 0.0; exttype = 0; initial = True

pwscfpy.pwpy_electrons_scf(printout, exxen, extpot, extene, exttype, initial)

etotal = pwscfpy.pwpy_calc_energies(extpot, extene, exttype)
print('etotal', etotal)

pwscfpy.pwpy_mod.pwpy_get_rho(rho)

ions = ase_io.ase_read(fname, format = 'espresso-in')
volume = ions.pos.cell.volume
fac = volume/np.size(rho)
sumd = np.sum(rho) * fac
print('ncharge : ', sumd)

pwpy_scf = pwscfpy.scf
rho = pwpy_scf.rho_core

pwscfpy.pwpy_forces(0)
forces = pwscfpy.force_mod.force.T
print('forces :', forces)
