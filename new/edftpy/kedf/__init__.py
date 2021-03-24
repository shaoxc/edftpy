import numpy as np
# from dftpy.kedf import KEDFunctional, KEDFStress
from dftpy.kedf import KEDF

from ..utils.common import AbsFunctional


class KEDFbak(AbsFunctional):
    def __init__(self, name = "WT", **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self, density, calcType=["E","V"], name=None, **kwargs):
        if name is None :
            name = self.name
        self.kwargs.update(kwargs)
        kwargs = self.kwargs
        return self.compute(density, name=name, calcType=calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], name=None, **kwargs):
        if name is None :
            name = self.name
        self.kwargs.update(kwargs)
        kwargs = self.kwargs
        functional = KEDFunctional(density, name = name, calcType = calcType, **kwargs)
        return functional