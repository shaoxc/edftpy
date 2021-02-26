from .mpi import GraphTopo
from dftpy.mpi import sprint as dftpy_sprint

__all__ = ["graphtopo", "sprint"]

graphtopo = GraphTopo()

def sprint(*args, comm = None, **kwargs):
    # kwargs['debug'] = True
    kwargs['flush'] = True
    comm = comm or graphtopo.comm
    dftpy_sprint(*args, comm = comm, **kwargs)
