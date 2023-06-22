from edftpy.utils.common import Functional
from .pseudopotential import LocalPP, ReadPseudo
from .xc import XC, get_libxc_names, get_short_xc_name
from .hartree import Hartree, hartree_energy
from .kedf import KEDF
from dftpy.ewald import ewald

class Ewald(ewald):
    def __init__(self, ions = None, rho = None, PME = True, **kwargs):
        obj = super().__init__(ions = ions, rho = rho, PME = PME, **kwargs)
        return obj
