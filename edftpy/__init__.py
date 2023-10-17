import os

__author__ = "Pavanello Research Group"
__contact__ = "m.pavanello@rutgers.edu"
__license__ = "MIT"
__version__ = "0.0.1rc0"
__date__ = "2023-10-17"

try:
    from importlib.metadata import version # python >= 3.8
except Exception :
    try:
        from importlib_metadata import version
    except Exception :
        pass

try:
    __version__ = version("edftpy")
except Exception:
    pass

if os.environ.get('EDFTPY_LOGLEVEL', None) is not None :
    os.environ['DFTPY_LOGLEVEL'] = os.environ.get('EDFTPY_LOGLEVEL')
