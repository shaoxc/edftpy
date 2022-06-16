.. _optimize:
   
========================
Optimization of Density
========================

Input Files
-----------

.. note::
   To avoid mixing some files, we strongly recommend setup each calculation (e.g. Optimization of density, relaxation of a structure of an MD simulation) in separate folders.

System Coordinates
~~~~~~~~~~~~~~~~~~

To carry out any calculation in the eDFTpy package you required a file with the .xyz coordinates, which includes the cell dimension of the system, the form of this file is a typical coordinate file text with an additional line placed before the first atom coordinate which includes the a,b,c lattice cell vectors. 
For this tutorial, we will use a water dimer coordinates with a vector cell lattice, `a = b = c = 12.0 Å`.

.. dropdown:: h2o_2.xyz

   .. literalinclude:: /tutorials/sdft/optimize/files/h2o_2.xyz

Pseudopotential files
~~~~~~~~~~~~~~~~~~~~~

Before carrying out any calculation on eDFTpy you should provide the pseudopotential files for all the chemical species in your system, Look for them in the GBRV_ pseudopotential site “QE” version, in this tutorial, user should provide the pseudopotentials for Oxigen: o_pbe_v1.2.uspp.F.UPF_ and Hydrogen: h_pbe_v1.4.uspp.F.UPF_ atomic species.

.. _GBRV: https://www.physics.rutgers.edu/gbrv/
.. _o_pbe_v1.2.uspp.F.UPF: https://www.physics.rutgers.edu/gbrv/o_pbe_v1.2.uspp.F.UPF
.. _h_pbe_v1.4.uspp.F.UPF: https://www.physics.rutgers.edu/gbrv/h_pbe_v1.4.uspp.F.UPF

eDFTpy Input File
~~~~~~~~~~~~~~~~~

Taking into account that in this tutorial we are dealing with a global system in which subsystems havethe same chemical composition, we will use the automatic definition of subsystems implemented in eDFTpy which we discuss later in this tutorial.

.. dropdown:: input.ini

   .. literalinclude:: /tutorials/sdft/optimize/files/input.ini

You can use this input.ini file to configure your SCF calculation. In general, this file is divided into six different sections describe below

* :ref:`[JOB]<JOB>` 

This flag describes the type of calculation you are going to carry out. Most of the time: 
  
.. literalinclude:: /tutorials/sdft/optimize/files/input-comments.ini
    :lines: 2-2

* :ref:`[PATH]<PATH>` 

Here you should write the location of your pseudopotential files (pp) and your input structure (cell). User can use the abbreviate notation for the location, we assume these files will be locating in the same folder that the input.ini

.. literalinclude:: /tutorials/sdft/optimize/files/input-comments.ini
    :lines: 5-6

* :ref:`[PP]<PP>` 
  
This section list the chemical species involve in the coordinate file and the name of speudopotential file that you will use for each atom. In the case of water dimer:
 
.. literalinclude:: /tutorials/sdft/optimize/files/input-comments.ini
    :lines: 9-10 

* :ref:`[OPT]<OPT>` 

This flag defines the charge density optimization of global system, in this tutorial we are going to use a maximun number of steps for optimization of 200 and the global system converged once the energy difference for the last 2 steps < `1e-6 a.u/atom`, using:
   
.. literalinclude:: /tutorials/sdft/optimize/files/input-comments.ini
    :lines: 13-14

* :ref:`[GSYSTEM]<GSYSTEM>` 
  
Structure parameters for the global system: 
  
.. literalinclude:: /tutorials/sdft/optimize/files/input-comments.ini
    :lines: 17-23

* :ref:`[SUB_KS]<SUB--prefix>` 

Subsystem configuration for the density optimization 
    
.. literalinclude:: /tutorials/sdft/optimize/files/input-comments.ini
    :lines: 26-29

Taking into account that we are going to use the automatic method to define the number of subsystems in the global system, we should include the method that we will use to determine each subsystem, in this case, we use the :ref:`distance<decompose-method>` means the program will base on the distance of atoms to decompose the subsystem when we used this method we should also include the radius of each element.

.. literalinclude:: /tutorials/sdft/optimize/files/input-comments.ini
    :lines: 30-34

.. note::
    Atomic radius value should be determine by the user.

Running eDFTpy
--------------

Once you setup your input file properly, you can run your claculation typing::

    $ mpirun -n 2 python -m edftpy input.ini

.. warning::
    * If you are using python3 version you should change all the python command lines wrote here from python to python3.
    
    * eDFTpy input file provided in this tutorial assumes that all your files including pseudopotentials are stored in the same working directory.

.. note::
    The number after *n* defines the number of cores you will use to perform your calculation.

Output Files 
------------

Once you run your calculation on eDFTpy the program  will generates a series of output files clasify between four different categories:

Outputs for Check 
~~~~~~~~~~~~~~~~~

  * `edftpy_running.json` Extended input file which contains all the calculation setup 
  * `edftpy_gsystem.xyz` Coordinates file for the output strcuture

Outputs of subsytem driver 
~~~~~~~~~~~~~~~~~~~~~~~~~~

  * `sub_prefix.xyz` Output coordinates for each subsystem driver 
  * `sub_prefix.out` Output file of each susbsytem driver

Temporary files of Subsystem driver 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  * `sub_ks_prefix.tmp` Temporary folder of each subsystem driver

Other properties files defined in input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  * Density of global system Saved in the `total.xsf` 
  * Density of each subsystem(s) Saved in `sub_prefix.xsf`
