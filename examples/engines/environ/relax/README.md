# Environ Engine

`EngineEnviron` is a engine for [`Environ`](https://gitlab.com/environ-developers/Environ) (package name is [`pyec`](https://gitlab.com/environ-developers/PyE-C)).

## Requirements
 - [Environ](https://gitlab.com/environ-developers/Environ) (latest)
 - [PyE-C](https://gitlab.com/environ-developers/PyE-C) (latest)
 - [QEpy](https://gitlab.com/shaoxc/qepy) (latest)

## Run

### Test Environ with sDFT
   The number of processors should be large than the number of subsystems. This example has one KS subsystem and one Environ subsystem.

 ```
 mpirun -n 2 python test_relax_environ.py
 ```
