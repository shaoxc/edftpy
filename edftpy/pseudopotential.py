import os
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
import scipy.special as sp
from dftpy.pseudo import LocalPseudo


class LocalPP(LocalPseudo):
    def __init__(self, grid=None, ions=None, PP_list=None, PME=True, **kwargs):
        # obj = super().__init__(grid = grid, ions = ions, PP_list = PP_list, PME = PME, MaxPoints = 1500, **kwargs)
        obj = super().__init__(grid = grid, ions = ions, PP_list = PP_list, PME = PME, MaxPoints = 15000, **kwargs)
        # obj = super().__init__(grid = grid, ions = ions, PP_list = PP_list, PME = PME, MaxPoints = 150000, **kwargs)
        return obj
