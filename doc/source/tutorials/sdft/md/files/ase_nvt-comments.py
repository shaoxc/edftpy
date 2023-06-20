#!/usr/bin/env python3
import os
from edftpy.api.api4ase import eDFTpyCalculator
from edftpy.config import read_conf
from edftpy.interface import conf2init
from edftpy.mpi import sprint, pmi

############################## initial ##############################
conf = read_conf('./input.ini')
graphtopo = conf2init(conf, pmi.size > 0)
cell_file = conf["PATH"]["cell"] +os.sep+ conf['GSYSTEM']["cell"]["file"]
#-----------------------------------------------------------------------
import ase.io
import ase.md
from ase.io.trajectory import Trajectory
from ase import units
from ase.md.npt import NPT
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.andersen import Andersen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

atoms = ase.io.read(cell_file)                                                            # Name of the file from ASE will read the coordinates

T = 300                                                                                   # Base temperature at the MD simulation will be runnied.
# T *= units.kB

calc = eDFTpyCalculator(config = conf, graphtopo = graphtopo)
atoms.set_calculator(calc)

MaxwellBoltzmannDistribution(atoms, temperature_K = T, force_temp=True)                   # Initialize the velocities at the defined temperature base on the Maxwell Boltzmann distribution


#dyn = Andersen(atoms, 1.5 * units.fs, temperature_K = T, andersen_prob=0.02)
dyn = NVTBerendsen(atoms, 0.7 * units.fs, temperature_K = T, taut=0.2*1000*units.fs)      # Define the type of thermostat we will used for the simulation and the time step used, for this tutorial we used a dt = 0.7 fs.

step = 0
interval = 1


def printenergy(a=atoms):
    global step, interval
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    sprint(
        "Step={:<8d} Epot={:.5f} Ekin={:.5f} T={:.3f} Etot={:.5f}".format(
            step, epot, ekin, ekin / (0.7 * units.kB), epot + ekin
        )
    )
    step += interval


dyn.attach(printenergy, interval=1)

traj = Trajectory("md.traj", "w", atoms)                                                  # Name of the output trajectory file
dyn.attach(traj.write, interval=1)
dyn.attach(traj.write, interval=1)

nsteps = 10                                                                               # Number of steps you will run your simulation
# dyn.run(nsteps)

from ase.optimize.optimize import Dynamics
dyn.max_steps = dyn.nsteps + nsteps
for converged in Dynamics.irun(dyn):
    if os.path.isfile('edftpy_stopfile'): exit()

atoms = ase.io.read(cell_file) 
atoms = ase.io.read("md-1.traj")
traj = Trajectory("md.traj", "w", atoms)
traj = Trajectory("md.traj", "a", atoms) 
