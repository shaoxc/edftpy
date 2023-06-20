import numpy as np
from edftpy.engine.engine_qe import EngineQE
from edftpy.engine.engine_environ import EngineEnviron
from edftpy.mpi import sprint

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except Exception:
    comm = None

use_environ = True

def run_scf(engine_qe, engine_environ, rho, comm = None, nstep = 20):
    for i in range(nstep):
        engine_qe.scf()
        engine_qe.scf_mix()
        energy = engine_qe.get_energy(olevel=2)
        sprint(f'QEpy    energy : {energy}', comm=comm)
        if use_environ:
            environ_energy = engine_environ.get_energy(olevel=0)
            sprint(f'Environ energy : {environ_energy}', comm=comm)
            energy += environ_energy
            sprint(f'Total   energy : {energy}', comm=comm)
            engine_qe.get_rho(rho)
            engine_environ.scf(rho, True)
            extpot = engine_environ.get_potential()
            engine_qe.set_extpot(extpot)

        if engine_qe.check_convergence(): break

    forces = engine_qe.get_force(icalc=0)
    sprint('QEpy    forces : \n', forces, comm=comm)

    if use_environ:
        environ_forces = engine_environ.get_force()
        sprint('Environ forces : \n', environ_forces, comm=comm)
        forces += environ_forces
        forces -= np.mean(forces, axis=0)
        sprint('Total   forces : \n', forces, comm=comm)
    return energy, forces


pw_file = 'neutral_2.in'

engine_qe = EngineQE()
engine_qe.initial(inputfile=pw_file, comm=comm)

if use_environ:
    inputs = EngineEnviron.get_kwargs_from_qepy()
    engine_environ = EngineEnviron()
    engine_environ.initial(comm=comm, **inputs)
    nr = engine_environ.get_grid()
    rho = np.zeros((np.prod(nr), 1), order='F')

energy, forces = run_scf(engine_qe, engine_environ, rho, comm)

if use_environ: engine_environ.stop_run()
engine_qe.stop_scf()
