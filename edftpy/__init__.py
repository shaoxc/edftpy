import os
from .version import __version__

if os.environ.get('EDFTPY_LOGLEVEL', None) is not None :
    os.environ['DFTPY_LOGLEVEL'] = os.environ.get('EDFTPY_LOGLEVEL')
