import numpy as np
from .utils.common import AbsFunctional

from dftpy import LibXC


class XC(AbsFunctional):
    def __init__(self, x_str='lda_x', c_str='lda_c_pz', **kwargs):
        pass

    def __call__(self, density, x_str='lda_x', c_str='lda_c_pz', calcType=["E","V"], **kwargs):
        return self.getFunctional(density, calcType=calcType, **kwargs)

    @classmethod
    def getFunctional(cls, density, x_str='lda_x', c_str='lda_c_pz', calcType=["E","V"], **kwargs):
        functional = LibXC(density=density, x_str=x_str, c_str = c_str, calcType = calcType, **kwargs)
        return functional
