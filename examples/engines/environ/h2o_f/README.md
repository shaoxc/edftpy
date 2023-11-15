# Environ Engine

`EngineEnviron` is a engine for [`Environ`](https://gitlab.com/environ-developers/Environ) (package name is [`pyec`](https://gitlab.com/environ-developers/PyE-C)).

## Requirements
 - [Environ](https://gitlab.com/environ-developers/Environ) (latest)
 - [PyE-C](https://gitlab.com/environ-developers/PyE-C) (latest)
 - [QEpy](https://gitlab.com/shaoxc/qepy) (latest)

## Run

### Test the engine with QEpy

 ```
 mpirun -n 1 python test_engine_environ.py
 ```

### Test Environ with sDFT
   The number of processors should be large than the number of subsystems. This example has two KS subsystems and one Environ subsystem.

 ```
 mpirun -n 3 python -m edftpy optim.ini
 ```
