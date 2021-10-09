# eDFTpy: Embedding & Multiscale Modeling in Python
A creation of PRG

`eDFTpy` combines Kohn-Sham Density Functional Theory (KS-DFT) and orbital-free DFT (OF-DFT) in an embedding framework, where only weak intermolecular interactions are treated at the OF-DFT level, while strong intramolecular interactions are treated at the KS-DFT level. The code is based on [`DFTpy`](http://dftpy.rutgers.edu) and [`QEpy`](https://gitlab.com/shaoxc/qepy) which have also been developed by [PRG](https://sites.rutgers.edu/prg/) at [Rutgers University-Newark](http://sasn.rutgers.edu).

## Requirements
 - Python 3.6 or later
 - [DFTpy](https://gitlab.com/pavanello-research-group/dftpy) (latest)
 - [Numpy](https://numpy.org/doc/stable)
 - [Scipy](https://docs.scipy.org/doc/scipy/reference)
 - [ASE](http://wiki.fysik.dtu.dk/ase)

### Optional (highly recommended):
 - [Libxc](https://gitlab.com/libxc/libxc)
 - [f90wrap](https://github.com/jameskermode/f90wrap) (latest)
