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
from ase import units
from ase.md.npt import NPT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

atoms = ase.io.read(cell_file)

T = 1023  # Kelvin
T *= units.kB

calc = eDFTpyCalculator(config = conf, graphtopo = graphtopo)
atoms.set_calculator(calc)

MaxwellBoltzmannDistribution(atoms, T, force_temp=True)

dyn = Langevin(atoms, 2 * units.fs, T, 0.1)

step = 0
interval = 1


def printenergy(a=atoms):
    global step, interval
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    sprint(
        "Step={:<8d} Epot={:.5f} Ekin={:.5f} T={:.3f} Etot={:.5f}".format(
            step, epot, ekin, ekin / (1.5 * units.kB), epot + ekin
        )
    )
    step += interval


dyn.attach(printenergy, interval=1)

traj = Trajectory("md.traj", "w", atoms)
dyn.attach(traj.write, interval=5)

dyn.run(500)
