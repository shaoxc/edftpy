import caspytep
import os
import ase.io.espresso as ase_io_driver
from ase.calculators.espresso import Espresso as ase_calc_driver
from collections import OrderedDict

from dftpy.constants import LEN_CONV

from edftpy.io import ions2ase
from edftpy.engine.engine import Engine

try:
    __version__ = caspytep.__version__
except Exception :
    __version__ = '0.0.1'

class EngineCastep(Engine):
    pass
