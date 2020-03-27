# Collection of Kinetic Energy Density Functionals
import numpy as np
from dftpy.kedf import KEDFunctional, KEDFStress

from ..utils.common import AbsFunctional


class KEDF(AbsFunctional):
    def __init__(self, name = "WT", **kwargs):
        self.name = name

    def __call__(self, density, calcType=["E","V"], name=None, **kwargs):
        if name is None :
            name = self.name
        return self.compute(density, name=name, calcType=calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], name=None, **kwargs):
        if name is None :
            name = self.name
        functional = KEDFunctional(density, name = name, calcType = calcType, **kwargs)
        return functional
