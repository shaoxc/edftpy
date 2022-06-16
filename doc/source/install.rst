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
* pylibxc_ (Exchange-correlation functionals)
* mpi4py_ 3.0.2 or newer (MPI for python)
* mpi4py-fft_ 2.0.4 or newer (Fast Fourier Transforms with MPI)
* xmltodict_ (For UPF pseudopotential (new version))
* upf_to_json_ (For UPF pseudopotential)

Optional:

* QEpy_ (to use Quantum ESPRESSO as KS-DFT engine (others available upon request))
* pyFFTW_  (Fast Fourier Transform)

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _pylibxc: https://tddft.org/programs/libxc/
.. _pyFFTW: https://pyfftw.readthedocs.io/en/latest/
.. _ASE: https://gitlab.com/ase/ase
.. _DFTpy: https://gitlab.com/pavanello-research-group/dftpy
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _xmltodict: https://github.com/martinblech/xmltodict
.. _upf_to_json: https://github.com/simonpintarelli/upf_to_json
.. _QEpy: https://gitlab.com/shaoxc/qepy
.. _f90wrap: https://github.com/jameskermode/f90wrap


Installation from source
========================

Git:
----------

    You can get the source from gitlab like this::

        $ git clone https://gitlab.com/pavanello-research-group/edftpy.git
        $ python -m pip install ./edftpy

    or simpler::

        $ python -m pip install git+https://gitlab.com/pavanello-research-group/edftpy.git



.. note::

    Because ``eDFTpy`` still under active development, non-backward-compatible changes can happen at any time. Please, clone the lastest release often.

Step by Step Installation
=========================

Pavanello Reseach Group has developed and step by step installation guide which has the goal to help the new or non-experienced users with the installation of eDFTpy package tested in Ubuntu/Debian and CentOS/RedHat based systems. 

Would you like to check the installation guide? :ref:`Click Here<step_by_step>`

