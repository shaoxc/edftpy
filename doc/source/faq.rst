.. _faq:


==========================
Frequently Asked Questions
==========================

Installation
============

- **Q**: When do I need install `QEpy`?

  **A**: QEpy_ is highly recommended to install, especially when you want to run KS-DFT embedding calculations.


Running
=======

- **Q**: Error "*OpenBLAS blas_thread_init: pthread_create failed for thread 49 of 52: Resource temporarily unavailable*"

  **A**: `Try <https://github.com/xianyi/OpenBLAS/issues/1668#issuecomment-402728065>`_:

   .. code:: shell

    export OMP_NUM_THREADS=1
    export USE_SIMPLE_THREADED_LEVEL3=1


.. _QEpy: https://gitlab.com/shaoxc/qepy
