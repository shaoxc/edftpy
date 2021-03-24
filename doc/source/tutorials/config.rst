.. _config:

=====================
Script mode of eDFTpy
=====================

eDFTpy is a set of python modules. However, it can be executed by using the `edftpy` script which is generated at installation time. Here's a quick guide to the script's configuration dictionary, or `config`. 


.. list-table::

     * - `JOB`_
       - `PATH`_
       - `MATH`_
       - `PP`_
     * - `OPT`_
       - `GSYSTEM`_
       - `SUB_`_
       - 

.. warning:: 
    `PP`_ is a mandatory input (i.e., no default is avaliable for it).

.. note::
    Defaults work well for most arguments.

    When *Options* is empty, it can accept any value.


JOB
----------

    Control of the running job.

.. option:: task

    The task to be performed.
        *Options* : Optdensity

        *Default* : Optdensity

.. option:: calctype

    The property to be calculated.
        *Options* : Energy, Potential, Both, Force

        *Default* : Energy

PATH
----------


MATH
----------

    Some methods and techniques that make `eDFTpy` really fast.

.. option:: linearii

    Linear-scaling method to deal with Ion-Ion interactions (PME).
        *Options* : True, False

        *Default* : True

.. option:: linearie

    Linear-scaling method to deal with Ion-Electron interactions (PME).
        *Options* : True

        *Default* : True

PP
----------

    Pseudopotential file of each atom type.

        *e.g.*

        - *Al* = Al_lda.oe01.recpot
        - *Mg* = Mg_lda.oe01.recpot

OPT
----------

GSYSTEM
----------

SUB\_
----------
