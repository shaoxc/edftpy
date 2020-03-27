import numpy as np
from .utils.common import AbsFunctional


class Embed(AbsFunctional):
    def __init__(self, **kwargs):
        self.args = kwargs

    def __call__(self, densitydict, calcType=["E","V"], **kwargs):
        args = self.args
        args.update(kwargs)
        return self.compute(densitydict, calcType=calcType, **args)

    def compute(self, densitydict, calcType=["E","V"], **kwargs):
        args = self.args
        args.update(kwargs)
        results = {}
        for key, rho in densitydict :
            results[key] = self.getFunctional(rho, calcType=calcType, **args)
        return results
