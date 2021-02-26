import os
from edftpy.api.api4ase import eDFTpyCalculator
from edftpy.config import read_conf
from edftpy.interface import conf2init
from edftpy.mpi import graphtopo, sprint, pmi

############################## initial ##############################
# conf = read_conf('./of.ini')
conf = read_conf('./qe.ini')
conf2init(conf, pmi.size > 0)
cell_file = conf["PATH"]["cell"] +os.sep+ conf['GSYSTEM']["cell"]["file"]
#-----------------------------------------------------------------------

import ase.io
from ase.io.trajectory import Trajectory
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

atoms = ase.io.read(cell_file)

trajfile = 'opt.traj'

calc = eDFTpyCalculator(config = conf, graphtopo = graphtopo)
atoms.set_calculator(calc)

af = atoms
opt = SciPyFminCG(af, trajectory = trajfile)

opt.run(fmax = 0.01)

if graphtopo.is_root :
    traj = Trajectory(trajfile)
    ase.io.write('opt.vasp', traj[-1], direct = True, long_format=True, vasp5 = True)
