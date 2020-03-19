from abc import ABC, abstractmethod
import numpy as np
from .kedf import KEDF


class TotalEvaluator(object):
    def __init__(self, nonadditive = None, **kwargs):
        self.nonadditive = nonadditive
        self.funcDict = {}
        self.funcDict.update(kwargs)
        for key, evalfunctional in self.funcDict.items():
            if evalfunctional is None:
                del self.funcDict[key]

    def __call__(self, densityDict, calcType=["E","V"], **kwargs):
        return self.compute(densityDict, calcType, **kwargs)

    def compute(self, densityDict, calcType=["E","V"], **kwargs):
        total = None
        rho = densityDict['TOTAL']
        for key, evalfunctional in self.funcDict.items():
            obj = evalfunctional(rho, calcType)
            if total is None :
                total = obj
            else :
                total += obj

        if 'E' in calcType :
            energy = total.energy
        nad = {}
        results = {}
        results['TOTAL'] = total.copy()
        for denType, nadfunc in self.nonadditive.items():
            nad[denType] = {}
            for key, evalfunctional in nadfunc.items():
                nad[denType][key] = evalfunctional(densityDict[denType], calcType)
                if 'E' in calcType :
                    energy += nad[denType][key].energy

        nadfunc = self.nonadditive['TOTAL']
        for denType, rho in densityDict.items():
            if denType == 'TOTAL' :
                continue
            results[denType] = results[denType].copy()
            for key, evalfunctional in nadfunc.items():
                obj = evalfunctional(rho, calcType)
                results[denType] += nad[denType][key]
                results[denType] -= obj
                if 'E' in calcType :
                    energy -= obj.energy
                # results['TOTAL'] += nad[denType][key]
                # results['TOTAL'] -= obj
        if 'E' in calcType :
            results['TOTAL'].energy = energy
        return results
