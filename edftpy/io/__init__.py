import os
import sys
from functools import wraps

from dftpy.formats import ase_io
from dftpy.formats.io import read, write, read_system, read_density
from dftpy.formats.ase_io import ase_read, ase_write, ase2ions, ions2ase
from .pp_xml import PPXmlGPAW, PPXml

def print2file(fileobj = None):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if fileobj is None :
                fobj = args[0].fileobj.fileno()
            else :
                fobj = fileobj.fileno()
            stdout = os.dup(1)
            os.dup2(fobj, 1)
            results = function(*args, **kwargs)
            os.dup2(stdout, 1)
            return results
        return wrapper
    return decorator
