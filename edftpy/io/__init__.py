import os
import sys
from functools import wraps

from dftpy.formats import ase_io
from dftpy.formats.io import read, write, read_system, read_density
from dftpy.formats.ase_io import ase_read, ase_write, ase2ions, ions2ase

def print2file(fileobj = None):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if fileobj is None :
                fobj = args[0].fileobj
            elif isinstance(fileobj, str):
                fobj = open(fileobj, 'a')
            else :
                fobj = fileobj
            stdout = os.dup(1)
            if fobj is not None : os.dup2(fobj.fileno(), 1)
            results = function(*args, **kwargs)
            os.dup2(stdout, 1)
            if isinstance(fileobj, str): fobj.close()
            return results
        return wrapper
    return decorator
