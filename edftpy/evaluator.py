from .utils.common import AbsFunctional


class Evaluator(AbsFunctional):
    def __init__(self, **kwargs):
        self.funcdicts = {}
        self.funcdicts.update(kwargs)
        for key, evalfunctional in self.funcdicts.items():
            if evalfunctional is None:
                del self.funcdicts[key]

    def __call__(self, density, calcType=["E","V"], **kwargs):
        return self.compute(density, calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], **kwargs):
        results = None
        for key, evalfunctional in self.funcdicts.items():
            obj = evalfunctional(density, calcType)
            if results is None :
                results = obj
            else :
                results += obj
        return results


class SubEvaluator(AbsFunctional):
    def __init__(self, nonadditive, subfunctional, **kwargs):
        self.nonadditive = nonadditive
        self.funcdicts = subfunctional

    def __call__(self, density, calcType=["E","V"], totalfunc = None, **kwargs):
        return self.compute(density, calcType, totalfunc=totalfunc, **kwargs)

    def compute(self, density, calcType=["E","V"], totalfunc = None, **kwargs):
        results = None
        for key, evalfunctional in self.funcdicts.items():
            obj = evalfunctional(density, calcType)
            if results is None :
                results = obj
            else :
                results += obj
        for key, evalfunctional in self.nonadditive.items():
            obj = evalfunctional(density, calcType)
            results -= obj
        if totalfunc is not None :
            results += totalfunc
        return results


class TotalEvaluatorAll(object):
    def __init__(self, nonadditive = None, **kwargs):
        self.nonadditive = nonadditive
        self.funcdicts = {}
        self.funcdicts.update(kwargs)
        for key, evalfunctional in self.funcdicts.items():
            if evalfunctional is None:
                del self.funcdicts[key]

    def __call__(self, densityDict, calcType=["E","V"], **kwargs):
        return self.compute(densityDict, calcType, **kwargs)

    def compute(self, densityDict, calcType=["E","V"], **kwargs):
        total = None
        rho = densityDict['TOTAL']
        for key, evalfunctional in self.funcdicts.items():
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
