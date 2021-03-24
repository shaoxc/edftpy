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

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _pylibxc: https://tddft.org/programs/libxc/
.. _pyFFTW: https://pyfftw.readthedocs.io/en/latest/
.. _ASE: https://gitlab.com/ase/ase
.. _DFTpy: https://gitlab.com/pavanello-research-group/dftpy
.. _f90wrap: https://github.com/jameskermode/f90wrap


Installation from source
========================


Git clone:
----------

    You can get the source from gitlab like this::

        $ git clone https://gitlab.com/pavanello-research-group/edftpy
        $ python setup.py install --user


.. note::

    Because ``eDFTpy`` still under active development, non-backward-compatible changes can happen at any time. Please, clone the lastest release often.
