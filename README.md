# eDFTpy: Multiscale Modeling in Python
This is firstly created at PRG March Hackathon.

`eDFTpy` combine the Kohn-Sham Density Functional Theory (KS-DFT) and orbital-free DFT (OF-DFT) in an embedding framework, that distills the best of both worlds whereby only weak intermolecular interactions are left to OF-DFT, and strong intramolecular interactions are left to KS-DFT. The code is based on [`DFTpy`](http://dftpy.rutgers.edu) which is developed by [PRG](https://sites.rutgers.edu/prg/) at [Rutgers University-Newark](http://sasn.rutgers.edu).

## Requirements
 - Python 3.6 or later
 - [DFTpy](https://gitlab.com/pavanello-research-group/dftpy) (latest)
 - [Numpy](https://numpy.org/doc/stable)
 - [Scipy](https://docs.scipy.org/doc/scipy/reference)
 - [ASE](http://wiki.fysik.dtu.dk/ase)

### Optional (highly recommended):
 - [Libxc](https://gitlab.com/libxc/libxc)
 - [f90wrap](https://github.com/jameskermode/f90wrap) (latest)
