.. _relax:
   
=======================
Relaxation of Structure
=======================

This tutorial will guide the user through the general workflow to perform a relaxation of a water dimer by using python based packages such as eDFTpy and the Atomic Simulation Environment (ASE_).

.. _ASE: https://gitlab.com/ase/ase

.. note::
   To avoid mixing some files, we strongly recommend setup each calculation (e.g. Optimization of density, relaxation of a structure of an MD simulation) in separate folders.

Input files
-----------

To carry out the relaxation of a structure, you have a set of input files divided in four different types: 

* Global system coordinates  
* eDFTpy input file  
* Aditional subsystem configuration
* ASE input file

Additional to these input files and the same that for the previous SCF tutorials, you also need the pseudopotential files for each of the atomic species described in your global system coordinates input file.

Global System Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~
In this tutorial, we are going to use the same global system structure provided in the second optimization of densitytutorial composed of a binary salt (NaCl) molecule surrounded by a water dimer.

.. dropdown:: salt_h2o_2.xyz

   .. literalinclude:: /tutorials/sdft/relax/files/salt_h2o_2.xyz

eDFTpy input file 
~~~~~~~~~~~~~~~~~

This input file follows the same structure that in the SCF Calculation with heterogeneous subsystems tutorial with some additional keywords in the Chloride subsystem section with the goal of setup the use of gaussian smearing to describe the occupations explained below:

.. dropdown:: input.ini

   .. literalinclude:: /tutorials/sdft/relax/files/input.ini

* :ref:`[SUB_CL]<SUB--prefix>`

.. literalinclude:: /tutorials/sdft/relax/files/input-comments.ini
    :lines: 45-46

Aditional subsystem configuration 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We defined a base file for the Chloride subsystem in the eDFTpy input file, we should also provide this input file with a Quantum Espresso input format. The goal of this input file is to provide an additional calculation setup for the  determined subsystem such as LDA+U (see the additional QEpy_ configuration to enable this type of calculation), which means you  can use one base file for each subsystem if you require and specify it in your eDFTpy input file.

.. dropdown:: qe_in_cl.in  

   .. literalinclude:: /tutorials/sdft/relax/files/qe_in_cl.in 

.. _QEpy: http://qepy.rutgers.edu/

ASE input file
~~~~~~~~~~~~~~

The goal of this input file is to configure the relaxation of a strcuture driven by ASE_ software, merging the subsystem configuration provide by the eDFTpy driver calculation. This input file follows the ASE_ input format for a relaxation calculation. 

.. dropdown:: ase_relax.py 

   .. literalinclude:: /tutorials/sdft/relax/files/ase_relax.py

To setup your own calculation you should take into account and reconfigure the next keywords sections:

.. literalinclude:: /tutorials/sdft/relax/files/ase_relax-comments.py
    :lines: 11-11
.. literalinclude:: /tutorials/sdft/relax/files/ase_relax-comments.py
    :lines: 34-34
.. literalinclude:: /tutorials/sdft/relax/files/ase_relax-comments.py
    :lines: 37-37

.. note::
   In this tutorial we use the limited memory version of the BFGS algorithm, check ASE_ website to use a different algorithm. On the other hand. we use a big value for `fmax`, you should change for a real calculation setup.

Running eDFTpy
--------------

Once you finish setting out the calculation, you can run the calculation using::

   $ mpirun -n 4 python -m ase_relax.py > log

.. note::
   Here the log file storages the summary of the calculation.

.. warning::
    If you are using python3 version you should change all the python command lines wrote here from python to python3

Output files
------------
Once your calculation finishes you will have the same set of output files from eDFTpy calculations with additional files as a product of the relaxation of the global system.

Additional output Files
~~~~~~~~~~~~~~~~~~~~~~~

  * `edftpy_gsystem.xyz` Global system coordinates for the last step of the simulation
  * `opt.xyz`  Coordinates for the global system optimized structure
  * `opt.traj` Trajectory file for the relaxation proccess

Preliminary Analysis
--------------------

Once you finished your simulation you would like to analyze your results, most of the time your structural analysis will require your trajectory file (opt.traj). Although this trajectory formatis only useful if you are using the ASE package to perform visual analysis.

Open the trajectory file with the ASE visualization tool by typing::

  $ ase gui opt.traj

Once you open the relaxation trajectory file using the visualization tool included in ASE package you will observe the step by step structure change and a graphic generator which includes a set of graphics check ASE's GUI_. Here we show the variation of the total energy (eV) in function of the number steps for two different relaxations of the  binary salt (NaCl) molecule surrounded by a water dimer, using the `fmax = 0.3` value provided in this tutorial (left) and a `fmax = 0.05` (Right).

.. |logo1| image:: /tutorials/sdft/relax/files/energy.png
    :scale: 80%

.. |logo2| image:: /tutorials/sdft/relax/files/energy-0.05.png
    :scale: 75%

.. table:: Total energy Variation
   :align: center

   +---------+---------+
   | |logo1| | |logo2| |
   +---------+---------+

.. _GUI: https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html

Convert the format of a trajectory file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you would like to use any other package to carry out structural analysis will be useful for you to covert this trajectory file to a most common type as xyz. We developed a python script that you can use to convert your trajectory files by typing::

$ python -m edftpy –-convert –-traj opt.traj -o opt-traj.xyz

You can check the command line :ref:`miscellaneous<cui>` for eDFTpy in the command line section of the sofware website.

Restarting a Relaxation of a structure
--------------------------------------

Restarting a calculation could be a powerful option taking into account the time-consuming disadvantage of an ab-initio molecular dynamics simulation. Here we give the user a guide on how can restart a relaxation that for a pecific reason did not finish or should be elongated.

To restart a relaxation calculation the user should modify the eDFTpy and ASE input files either Structure optimizartion or MD production, we suggest the next six-step workflow to have succes restarting a calculation.

#. Copy your trajectory file (opt.traj) to a new file::

   $ cp opt.traj  opt-1.traj

#. Create a new global system coordinate file using the last geometry coordinates::

   $ cp edftpy_gsystem.xyz rst-1.xyz

#. Change the file name of the cell-file in the eDFTpy input file
   
   * from
     
   .. literalinclude:: /tutorials/sdft/relax/files/input-comments.ini
      :lines: 24-24

   * to
    
   .. literalinclude:: /tutorials/sdft/relax/files/input-comments.ini
      :lines: 25-25

#. Change the cell file in the ASE input file to your old trajectory file
   
   * from
     
   .. literalinclude:: /tutorials/sdft/relax/files/ase_relax-comments.py
     :lines: 27-27

   * to
    
   .. literalinclude:: /tutorials/sdft/relax/files/ase_relax-comments.py
      :lines: 42-42

#. Restart your simulation using::

   $ mpirun -n 4 python -m ase_relax.py > log

#. Once you fininsh your structure optimization after one of many restartings you could concatenate the trajectory files using::

   $ python -m edftpy --convert --traj opt-1.traj opt-2.traj -o opt-tot.traj
