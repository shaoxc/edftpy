__author__ = "Pavanello Research Group"
__contact__ = "m.pavanello@rutgers.edu"
__license__ = "MIT"
__version__ = "0.0.1"
__date__ = "2021-09-08"

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
