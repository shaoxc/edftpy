__author__ = "Pavanello Research Group"
__contact__ = "m.pavanello@rutgers.edu"
__license__ = "MIT"
__version__ = "0.0.1"
__date__ = "2021-09-08"

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("edftpy")
except PackageNotFoundError:
    pass
