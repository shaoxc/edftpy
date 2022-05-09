.. _cui:

.. highlight:: bash

=============
Command lines
=============

eDFTpy also has some tools:

==============  =========================================================
tool            description
==============  =========================================================
help            Help for tools
run             Run the calculation with input file (Default)
convert         Convert the formats of files
==============  =========================================================


Help
====

For all tools, you can do::

    $ python -m edftpy -h
    $ python -m edftpy --convert -h
    $ python -m edftpy.cui.convert -h


.. _RUN:

Run
===

:ref:`config`::

    $ python -m edftpy edftpy.ini --mpi
    $ python -m edftpy --run edftpy.ini --mpi

.. _CONVERT:

Convert
=======

Convert structure formats::

    $ python -m edftpy --convert gsystem.xyz -o gsystem.vasp --frac
    $ python -m edftpy --convert gsystem.snpy -o gsystem.xsf
    $ python -m edftpy --convert edftpy_running.json -o gsystem.vasp

Concatenate the ASE trajectory files::

    $ python -m edftpy --convert --traj a.traj b.traj -o add.traj

Subtract subsystems from global system::

    $ python -m edftpy --convert --subtract gsystem.xyz sub_0.xyz -o rest.xyz
    $ python -m edftpy --convert --subtract gsystem.snpy sub_0.snpy -o rest.snpy

If the shape of two densities are different, will interpolate the second to the first one::

    $ python -m edftpy --convert --subtract sub_0.xsf sub_1.xsf -o diff.xsf

Add subsystems to global system::

    $ python -m edftpy --convert sub_0.xyz sub_1.xyz -o gsystem.xyz
    $ python -m edftpy --convert sub_*.snpy -o gsystem.snpy
    $ python -m edftpy --convert --format-in='espresso-in' sub_*.in -o gsystem.vasp

If the size of subsystems are different from global system, should also provide the json file::

    $ python -m edftpy --convert --json='edftpy_running.json' sub_ks_*.snpy -o total.xsf
    $ python -m edftpy --convert --json='edftpy_running.json' --subtract total.xsf sub_ks_0.snpy -o sub_rest.xsf

Get the density from QE output (with QEpy)::

    $ python -m edftpy --convert --qepy sub_ks.in -o sub_ks.xsf

Write a structure file base on a basefile (QE or CASTEP)::

    $ python -m edftpy --convert --basefile='qe.in' --format-out='espresso-in' gsystem.vasp -o gsystem.in
    $ python -m edftpy --convert --basefile='castep.cell' --format-out='castep' gsystem.vasp -o gsystem.cell
