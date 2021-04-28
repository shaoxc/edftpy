.. _download_and_install:

============
Installation
============

Requirements
============

* Python_ 3.6 or newer
* NumPy_ 1.8.0 or newer
* SciPy_ 0.18 or newer
* DFTpy_ latest
* ASE_  3.21.1 or newer

Optional:

* pylibxc_ (Exchange-correlation functionals other than LDA)
* pyFFTW_  (Fast Fourier Transform)
* f90wrap_ (F90 to Python interface generator with derived type support)
* mpi4py_ (MPI for python)
* mpi4py-fft_ (Fast Fourier Transforms with MPI)
* xmltodict_ (For UPF pseudopotential)
* upf_to_json_ (For UPF pseudopotential)
* QEpy_ (to use Quantum ESPRESSO as KS-DFT engine (others available upon request))

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _pylibxc: https://tddft.org/programs/libxc/
.. _pyFFTW: https://pyfftw.readthedocs.io/en/latest/
.. _ASE: https://gitlab.com/ase/ase
.. _DFTpy: https://gitlab.com/pavanello-research-group/dftpy
.. _f90wrap: https://github.com/jameskermode/f90wrap
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _xmltodict: https://github.com/martinblech/xmltodict
.. _upf_to_json: https://github.com/simonpintarelli/upf_to_json
.. _QEpy: https://gitlab.com/shaoxc/qepy


Installation from source
========================


Git clone:
----------

    You can get the source from gitlab like this::

        $ git clone https://gitlab.com/pavanello-research-group/edftpy
        $ python setup.py install --user


.. note::

    Because ``eDFTpy`` still under active development, non-backward-compatible changes can happen at any time. Please, clone the lastest release often.
