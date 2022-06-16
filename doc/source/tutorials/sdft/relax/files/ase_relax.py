#!/usr/bin/env python3

import os
import numpy as np
from edftpy.api.api4ase import eDFTpyCalculator
from edftpy.config import read_conf
from edftpy.interface import conf2init
from edftpy.mpi import graphtopo, sprint, pmi
np.random.seed(8888)
############################## initial ##############################
conf = read_conf('./input.ini')
conf2init(conf, pmi.size > 0)
cell_file = conf["PATH"]["cell"] +os.sep+ conf['GSYSTEM']["cell"]["file"]
#-----------------------------------------------------------------------

import ase.io
from ase.io.trajectory import Trajectory
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.constraints import FixAtoms
from ase.md.andersen import Andersen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixBondLength
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

atoms = ase.io.read(cell_file)

calc = eDFTpyCalculator(config = conf, graphtopo = graphtopo)
atoms.set_calculator(calc)

trajfile = 'opt.traj'
af = atoms
opt = LBFGS(af, trajectory = trajfile, memory = 10, use_line_search = False)
# opt = SciPyFminCG(af, trajectory = trajfile)

opt.run(fmax = 0.3)

traj = Trajectory(trajfile)
ase.io.write('opt.xyz', traj[-1])
