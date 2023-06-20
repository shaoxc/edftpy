.. _optimize-2:

======================================================
Optimization of Density with Heterogeneous susbsystems
======================================================

This tutorial assumes the user has already finished the :ref:`Optimization of Density<optimize>` tutorial, using the automatic definition of subsystems when you have the same chemical nature for each subsystem or have previous experience setting up QE calculations. This tutorial will focus on the details, to run a calculation that requires including a manual definition of subsystems combined with the adaptive definition of subsystems to run an SCF calculation of a global system composed of a binary salt (NaCl) molecule surrounded by a water dimer. 

.. note::
   To avoid mixing some files, we strongly recommend setup each calculation (e.g. Optimization of density, relaxation of a structure of an MD simulation) in separate folders.

Input Files
-----------

Following the input description provided in the previous SCF Calculation for a water dimer, you also requiere the global system coordinates and the eDFTpy input file.

.. dropdown:: salt_h2o_2.xyz

   .. literalinclude:: /tutorials/sdft/optimize-2/files/salt_h2o_2.xyz

.. dropdown:: input.ini

   .. literalinclude:: /tutorials/sdft/optimize-2/files/input.ini

This eDFTpy input file includes the same section’s flags with the addition of some keywords and sections, as described below:

* :ref:`[PP]<PP>` 

Addition of Sodium(I) and Chloride pseudopotentials
  
.. literalinclude:: /tutorials/sdft/optimize-2/files/input-comments.ini
    :lines: 9-12

* :ref:`[MOL]<MOL>`

This new section includes the charge of the molecules and atoms, note that in this tutorial this section is mandatory due to the presence of charge subsystems, this section should describe the total type of molecules listed in the subsystems section. 

.. literalinclude:: /tutorials/sdft/optimize-2/files/input-comments.ini
    :lines: 19-21

.. note:: 
    eDFTpy only can simulate neutral globral systems.

* :ref:`[SUB_NA]<SUB--prefix>` 

Input definition of the Sodium (I) subsystem 

.. literalinclude:: /tutorials/sdft/optimize-2/files/input-comments.ini
    :lines: 34-40

* :ref:`[SUB_ CL]<SUB--prefix>`
 
Input definition of the Chloride subsystem 

.. literalinclude:: /tutorials/sdft/optimize-2/files/input-comments.ini
    :lines: 44-50

* :ref:`[SUB_ H2O]<SUB--prefix>` 

Water subsystems will be defined using the same automatic definition based on the average radii starting at the third atom position of the `cell_file` 

.. literalinclude:: /tutorials/sdft/optimize-2/files/input-comments.ini
    :lines: 57-62

.. note:: 
    Unlike the basic SCF tutorial in this input parameter, we also include a mixing parameter for the water subsystem to keep consistency with the other subsystems, you can use different mixing parameter values for each subsystem if you require it. 

Running eDFTpy
--------------

Type the command below to run the calculation set up before:: 

$ mpirun -n 4 python -m edftpy input.ini

.. warning:: 
    To carry out this tutorial you are required to have a computer with at least four (4) available cores of processing, if you would prefer you can use a bigger number of cores if they are available in your computer unit.


Output Files 
------------

This calculation will generate the same categories of outputs discussed in the SCF base tutorial with additional files for the ions subsystems following:

Outputs to Check 
~~~~~~~~~~~~~~~~

  * `edftpy_running.json` Extended input file which contains all the calculation setup 
  * `edftpy_gsystem.xyz` Coordinates file for the output strcuture

Outputs of subsytem driver 
~~~~~~~~~~~~~~~~~~~~~~~~~~

  * `sub_prefix.xyz` Output coordinates for each subsystem driver 
  * `sub_prefix.out` Output file of each susbsytem driver

Temporary files of Subsystem driver 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  * `sub_prefix.tmp` Temporary folder of each subsystem driver

Other properties files defined in input file 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  * Density of global system Saved in the `total.xsf`
  * Density of each subsystem(s) Saved in  `sub_prefix.xsf`
