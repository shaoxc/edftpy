.. _md:

=============================
Molecular Dynamics Simulation
=============================

This tutorial will guide the user through the general workflow to perform a molecular dynamicsi (MD) simulation for a water dimer by using python based packages such as eDFTpy and the Atomic Simulation Environment (ASE)

.. note::
   To avoid mixing some files, we strongly recommend setup each calculation (e.g. Optimization of density, relaxation of a structure of an MD simulation) in separate folders.

Input Files
-----------

Same as the relaxation  step, MD simulation also, requires a set of input files divided into four different types:
 
 * Global system coordinates
 * eDFTpy input file
 * Aditional subsystem configuration
 * ASE input file

Global system coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~
In this tutorial, we are going to use the optimized structure obtaned as an output from the relaxation of a structure tutorial `opt.xyz` however, this workflow in real conditions depends on the needs of each user, this means there may be times when the user does not have to perform a relaxation calculation prior to the molecular dynamics simulation.

.. dropdown:: opt.xyz

   .. literalinclude:: /tutorials/sdft/md/files/opt.xyz

eDFTpy input file
~~~~~~~~~~~~~~~~~

Essentially, this file is the same as the one provided in the previous step. Nevertheless, for this step of the tutorial, you should change the cell file name to `opt.xyz`. Also, we  are going to use the adaptive definition of the subsystems for the global system through the MD simulation, using the same distance method introduced in the SCF tutorial with the automatic definition of subsystems. It means we should include the decompose keyword and all of the atomic radius in the :ref:`[GSYSTEM]<GSYSTEM>` flag, as follows:

.. dropdown:: input.ini

   .. literalinclude:: /tutorials/sdft/md/files/input.ini

* :ref:`[GSYSTEM]<GSYSTEM>`

.. literalinclude:: /tutorials/sdft/md/files/input-comments.ini
    :lines: 24-24

.. literalinclude:: /tutorials/sdft/md/files/input-comments.ini
    :lines: 30-34

Aditional subsystem configuration 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the same base file provided for the relaxation step, you can change to different parameters if you require.

.. dropdown:: qe_in_cl.in

   .. literalinclude:: /tutorials/sdft/relax/files/qe_in_cl.in

ASE input file 
~~~~~~~~~~~~~~

Followed the ASE input file explanation for the relaxation of a strcutre, this input file also is written following the ASE input file format. Contrary to the previous file, this input file includes definitions of temperature, thermodynamics ensemble, thermostat, time step, etc used to run the simulation, follows we summarize the ones that we consider you should think about before setting up a real calculation. Check ASE_ Molecular dynamics module for an extensive keywords description.

.. dropdown:: ase_nvt.py 

   .. literalinclude:: /tutorials/sdft/md/files/ase_nvt.py


.. literalinclude:: /tutorials/sdft/md/files/ase_nvt-comments.py
    :lines: 23-25
.. literalinclude:: /tutorials/sdft/md/files/ase_nvt-comments.py
    :lines: 31-31
.. literalinclude:: /tutorials/sdft/md/files/ase_nvt-comments.py
    :lines: 35-35
.. literalinclude:: /tutorials/sdft/md/files/ase_nvt-comments.py
    :lines: 55-55
.. literalinclude:: /tutorials/sdft/md/files/ase_nvt-comments.py
    :lines: 59-59

.. _ASE: https://wiki.fysik.dtu.dk/ase/ase/md.html

Running eDFTpy
--------------

Once you finish setting out the calculation, you can run the calculation using::

$ mpirun -n 4 python -m ase_relax.py > log

.. note::
   Here the log file storages the summary of the calculation.

.. warning::
    If you are using python3 version you should change all the python command lines wrote here from python to python3

Additional output Files
-----------------------
  * `edftpy_gsystem.xyz` Global system coordinates for the last step of the simulation
  * `md.traj` Output trajectory file for the global system.

Preliminary Analysis 
--------------------

Once you finished your simulation you would like to analyze your results, most of the time your structural analysis will require your trajectory file (md.traj). Although this trajectory formatis only useful if you are using the ASE package to perform visual analysis.

Open the trajectory file with the ASE visualization tool by typing::

  $ ase gui md.traj

Once you open MD trajectory file observe the step by step structure change and a graphic generator which includes a set of graphics check ASE's GUI_. Here we show the variation of the total energy (eV) in function of the number steps for the MD simulation of the  binary salt (NaCl) molecule surrounded by a water dimer by 20 steps of simulation. 

.. |logo1| image:: /tutorials/sdft/md/files/energy-10.png
    :scale: 80%

.. table:: Total energy Variation
   :align: center

   +---------+
   | |logo1| | 
   +---------+

.. note::
   For the purposes of this tutorial, the configured input file only includes 20 steps, otherwise the wait time would be proportional to a couple of days using four (4) cores.

.. _GUI: https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html

Convert the format of a trajectory file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you would like to use any other package to carry out structural analysis will be useful for you to covert this trajectory file to a most common type as xyz. We developed a python script that you can use to convert your trajectory files by typing::

$ python -m edftpy –-convert –-traj md.traj -o md-traj.xyz

You can check the command line :ref:`miscellaneous<cui>` for eDFTpy in the command line section of the sofware website.

Restarting a MD simulation
--------------------------

Restarting a calculation could be a powerful option taking into account the time-consuming disadvantage of an ab-initio molecular dynamics simulation. Here we give the user a guide on how can restart an MD simulation

To restart a simulation the user should modify the eDFTpy and ASE input files either Structure optimizartion or MD production.

#. Copy your trajectory file (md.traj) to a new file::
   
   $ cp md.traj md-1.traj

#. Create a new global system coordinate file using the last geometry coordinates::
   
   $ cp edftpy_gsystem.xyz rst-1.xyz

#. Change the file name for the cell-file in the eDFTpy input file 

   * from

   .. literalinclude:: /tutorials/sdft/md/files/input-comments.ini
    :lines: 24-24

   * to
   .. literalinclude:: /tutorials/sdft/md/files/input-comments.ini
    :lines: 25-25

#. Change the `cell_file` in the ASE input file to your old trajectory file 
   
   * from

   .. literalinclude:: /tutorials/sdft/md/files/ase_nvt-comments.py
    :lines: 67-67
  
   * to

   .. literalinclude:: /tutorials/sdft/md/files/ase_nvt-comments.py
    :lines: 68-68

#. As you already initialize velocities from your las trajectory you should not include the Maxwell Boltzmann Distribution line, you can comment this line adding a hash `#` at he begining of the line

#. To avoid to concatenate all the trajectory file at the end of the simulation you can attempt the new trajectory to  the old `md.traj` file modifying the trajectory file section: 

   * from

   .. literalinclude:: /tutorials/sdft/md/files/ase_nvt-comments.py
    :lines: 69-69
  
   * to

   .. literalinclude:: /tutorials/sdft/md/files/ase_nvt-comments.py
    :lines: 70-70
   
#.  Restart your simulation using::

    $ mpirun -n 4 python -m ase_relax.py > log
