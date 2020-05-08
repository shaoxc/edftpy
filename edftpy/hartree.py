import numpy as np
from .utils.common import AbsFunctional
from dftpy.hartree import HartreeFunctional, HartreeFunctionalStress


class Hartree(AbsFunctional):
    def __init__(self, **kwargs):
        pass

    def __call__(self, density, calcType=["E","V"], **kwargs):
        return self.compute(density, calcType=calcType, **kwargs)

    @classmethod
    def compute(cls, density, calcType=["E","V"], **kwargs):
        functional = HartreeFunctional(density, calcType=calcType)
        return functional
