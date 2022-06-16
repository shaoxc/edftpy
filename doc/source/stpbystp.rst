.. _step_by_step:

===============================
Step by Step Installation guide
===============================

Before Starting
---------------

#. If you are going to perform the configuration from scratch, we advise carry out a clean intallation in you system, it means deactivate any virtual enviroment in your system (e.g. Anaconda, Miniconda) before performing the installation it will avoid any incompatibility between libraries.

#. Update your linux distribution
   
   * Ubuntu/Debian::

       $ sudo apt-get update && sudo apt-get upgrade

   * CentOS/RedHat::
     
       $ sudo yum check-update
       $ sudo yum update -y

#. Most of the Unix distributions already includes python packages, before starting you installation you should confirm your python package version. To do this you could type in your shell::

    $ python --version

#. If you have python version > 3.9 (e.g. 3.10.4 or later) you should install libxc from source due to the imcopatibility with the pip environment in which eDFTpy was created (see step 10 from Ubuntu based systems tutorial).

#. Due to eDFTpy and QEpy packages depends on Quantum Espresso=6.5, you must use gfortran and gcc compilers version <=9.0 (see step 11 from Ubuntu based systems tutorial).
   
#. If you have python version > 3.9 (e.g. 3.10.4 or later) you should install libxc from source due to the imcopatibility with the pip environment in which eDFTpy was created (see step 10 from Ubuntu based systems tutorial).

Ubuntu or Debian based OS
-------------------------

.. note::
   This tutorial was developed using Ubuntu 22.04 LTS Desktop distribution.

Prerequisites 
~~~~~~~~~~~~~

#. Install a CMake compiler::
   
   $ sudo apt-get install cmake

#. Install Make tool::
   
   $ sudo apt-get install make

#. Install PyPA tool for installing Python packages pip::
   
   $ sudo apt install python-pip

#. Install the distribute version control system Git::
   
   $ sudo apt-get install git-all

#. Install numpy::
   
   $ pip install numpy

#. Install the C subroutine library for computing the discrete Fourier transform (DFT) FFTW libraties. We strongly recommend performing the installation using the Linux package provider due to an unexpected incompatibility with the source file libraries installation::
   
   $ sudo apt install libfftw3-3 libfftw3-dev

#. Install gfortran and gcc compilers::
   
   $ sudo apt-get install gfortran
   $ sudo apt-get install gcc

#. Configure your parallel environment using Open MPI, tested version 4.1.4.
   
* Go to the Open MPI website_
   
* Download the release 4.1.4: Click on openmpi-4.1.4.tar.gz release
   
* Go to your Downloads folder::
          
   $ cd /home/user/Downloads
   
* Uncompress your tar.gz file::
           
   $ tar -xzvf openmpi-4.1.4.tar.gz
   
* Here you can follow the installation instructions provided in INSTALL file inside of your openmpi-4.1.4 folder
    
* Create and environmental variable in your ~/.bascrh adding the next strings::
   
   $ export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH 

.. note::
   This path could change if you performed your Open MPI installation in a different directory than the default one.

.. _website: https://www.open-mpi.org/ 
.. _Q.E: https://gitlab.com/QEF/q-e/-/releases/qe-6.5
.. _libxc: https://tddft.org/programs/libxc/
.. warning::
   If you compile Open MPI using the default location you should carry out the make command with sudo privileges.


Install Packages
~~~~~~~~~~~~~~~~

1. Install Libxc from source. Skip this step if your python version is < 3.10.4

* Go to libxc_ website 

* Download the libxc-5.2.3.tar.gz release

* Go to your Downloads folder: cd /home/user/Downloads

* Uncompress your tar.gz file: tar -xzvf libxc-5.2.3.tar.gz

* Compile Libxc::
  
    $ ./configure
    $ make
    $ make install
 
2. Change your gcc and gfortran version from the newest versions to 9.4.0.::
   
   $ sudo apt-get install gfortran-9
   $ sudo apt-get install gcc-9

3. Once you install the version 9 packages the easiest way to perform this change of compilers in your system is creating an alias string in your `~/.basrch`::

   $ alias gcc='gcc-9'
   $ alias gfortran='gfortran-9'

Here DO NOT omit the quotation marks.
   
4. Update your system variables::
   
   $ source ~/.bashrc

5. Install Quantum Espresso package

* Go to Q.E_. 6.5 release

* Download the Source code (`tar.gz`) file

* Uncompress your tar.gz file::
  
   $ tar -xzvf q-e-qe-6.5.tar.gz

* Go to your Quantum Espresso file: cd q-e-qe-6.5
* Make the configuration

.. note::
   All static libraries should be compiled with the -fPIC compuiler option. Add -fPIC to the configuration options.

* Parallel environment::
  
   $ ./configure  MPIF90=mpif90  --with-scalapack=no -enable-openmp=no -enable-parallel=yes   CFLAGS=-fPIC FFLAGS=-fPIC try_foxflags=-fPIC 
   $ make all
 
6. Install the eDFTpt package::
   
   $ git clone https://gitlab.com/pavanello-research-group/edftpy.git
   $ python -m pip install ./edftpy

7. Install the QEpy package::
   
   $ git clone --recurse-submodules https://gitlab.com/shaoxc/qepy.git
   $ qedir=/full/path/to/your/q-e-qe-6.5/ python -m pip install -U ./qepy

.. note::
   qedir cannot the abreviated, must be the full path to your QE folder.

.. warning::
   If you compile Open MPI using the default location you should carry out the make command with sudo privileges.

CentOS and RedHat based OS
--------------------------

.. note::
   This tutorial was developed using CentOS 7 Desktop distribution.

Prerequisites 
~~~~~~~~~~~~~

#. Configure your python3 environment::
   
   $ sudo yum install $https://repo.ius.io/ius-release-el$(rpm -E '%{rhel}').rpm$
   $ sudo yum update -y
   $ sudo yum install -y python3
   $ sudo yum install python3-devel.x86_64

#. Install a CMake compiler::
   
   $ sudo yum install cmake

#. Install Make tool::
  
   $ sudo yum install make

#. Install PyPA tool for installing Python packages pip::
   
   $ sudo yum install epel-release
   $ sudo yum install python3-pip

#. Install the distribute version control system Git::
   
   $ sudo yum install git

#. Install numpy::
   
   $ pip3 install numpy --user

#. Install the C subroutine library for computing the discrete Fourier transform (DFT) FFTW libraties. We strongly recommend performing the installation using the Linux package provider due to an unexpected incompatibility with the source file libraries installation::
 
   $ sudo yum -y install fftw-devel

#. Install gfortran and gcc compilers::
   
   $ sudo yum install gcc-gfortran
   $ sudo yum install gcc-g++

#. Install LAPACK libraries::

   $ sudo yum -y install lapack-devel

#. Install cython libraries::
   
   $ python3 -m pip install -U cython --user

#. Configure your parallel environment using Open MPI, tested version 4.1.4.
   
* Go to the Open MPI website_
   
* Download the release 4.1.4: Click on openmpi-4.1.4.tar.gz release
   
* Go to your Downloads folder::
          
   $ cd /home/user/Downloads
   
* Uncompress your tar.gz file::
           
   $ tar -xzvf openmpi-4.1.4.tar.gz
   
* Here you can follow the installation instructions provided in INSTALL file inside of your openmpi-4.1.4 folder
    
* Create and environmental variable in your ~/.bascrh adding the next strings::
   
   $ export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH 

.. note::
   This path could change if you performed your Open MPI installation in a different directory than the default one.

.. warning::
   If you compile Open MPI using the default location you should carry out the make command with sudo privileges.


Install Packages
~~~~~~~~~~~~~~~~

1. Install Libxc from source. Skip this step if your python version is < 3.10.4

* Go to libxc_ webpage

* Download the libxc-5.2.3.tar.gz release

* Go to your Downloads folder: cd /home/user/Downloads

* Uncompress your tar.gz file: tar -xzvf libxc-5.2.3.tar.gz

* Compile Libxc::
  
    $ ./configure
    $ make
    $ make install
 
2. Install Quantum Espresso package

* Go to Q.E_. 6.5 release

* Download the Source code (`tar.gz`) file

* Uncompress your tar.gz file::
  
   $ tar -xzvf q-e-qe-6.5.tar.gz

* Go to your Quantum Espresso file: cd q-e-qe-6.5
* Make the configuration

.. note::
   All static libraries should be compiled with the -fPIC compuiler option. Add -fPIC to the configuration options.

* Parallel environment::
  
   $ ./configure  MPIF90=mpif90  --with-scalapack=no -enable-openmp=no -enable-parallel=yes   CFLAGS=-fPIC FFLAGS=-fPIC try_foxflags=-fPIC 

3. Install the eDFTpy package::
   
   $ git clone https://gitlab.com/pavanello-research-group/edftpy.git
   $ python3 -m pip install ./edftpy --user

4. Install the QEpy package::
   
   $ git clone --recurse-submodules https://gitlab.com/shaoxc/qepy.git
   $ qedir=/full/path/to/your/q-e-qe-6.5/ python -m pip install -U ./qepy --user

.. note::
   qedir cannot the abreviated, must be the full path to your QE folder.
