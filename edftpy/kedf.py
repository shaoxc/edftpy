# Collection of Kinetic Energy Density Functionals
import numpy as np
from .utils.common import AbsFunctional

from dftpy.kedf import KEDFunctional, KEDFStress


class KEDF(AbsFunctional):
    def __init__(self, name = "WT", **kwargs):
        self.name = name

    def __call__(self, density, name=None, calcType=["E","V"], **kwargs):
        if name is None :
            name = self.name
        return self.getFunctional(density, name=name, calcType=calcType, **kwargs)

    @classmethod
    def getFunctional(cls, density, name=None, calcType=["E","V"], **kwargs):
        if name is None :
            name = cls.name
        functional = KEDFunctional(density, name, calcType, **kwargs)
        return functional
