from qepy.driver import Driver
from mpi4py import MPI
comm = MPI.COMM_WORLD
inputfile = 'sub_ho.in'

# Run scf
# driver = Driver(inputfile, comm)
# driver.scf()
# driver.save()
# Run TDDFT
driver = Driver(inputfile, comm, task = 'optical')
driver.scf()
dipole = driver.get_dipole_tddft()
driver.stop()
