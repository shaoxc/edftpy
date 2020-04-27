import numpy as np
from .utils.common import Field, Functional, AbsFunctional
from dftpy.functionals import FunctionalClass


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
        #-----------------------------------------------------------------------
        nc = np.sum(density)
        density[density < 1E-30] = 1E-30
        density *= nc/np.sum(density)
        #-----------------------------------------------------------------------
        for key, evalfunctional in self.funcdicts.items():
            obj = evalfunctional(density, calcType)
            # print(key, obj.energy * 27.21138)
            if results is None :
                results = obj
            else :
                results += obj
        if results is None :
            # results = Functional(name = 'NONE')
            if 'E' in calcType :
                energy = 0.0
            else :
                energy = None
            if 'V' in calcType :
                potential = Field(grid=density.grid, rank=1, direct=True)
            else :
                potential = None
            results = Functional(name = 'ZERO', energy=energy, potential=potential)
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


class EnergyEvaluatorMix(AbsFunctional):
    def __init__(self, embed_evaluator = None, sub_evaluator = None, sub_functionals = None, **kwargs):
        """
        sub_functionals : [dict{'type', 'name', 'options'}]
        """
        self.embed_evaluator = embed_evaluator
        self.sub_evaluator = sub_evaluator
        # self.sub_evaluator = self.get_sub_evaluator(sub_functionals)
        self._gsystem = None

    def get_sub_evaluator(self, sub_functionals, **kwargs):
        funcdicts = {}
        for item in sub_functionals :
            func = FunctionalClass(type=item['type'], name = item['name'], **item['options'])
            funcdicts[item['type']] = func
        sub_evaluator = Evaluator(**funcdicts)
        return sub_evaluator

    @property
    def gsystem(self):
        if self._gsystem is not None:
            return self._gsystem
        else:
            raise AttributeError("Must specify gsystem for EnergyEvaluatorMix")

    @gsystem.setter
    def gsystem(self, value):
        self._gsystem = value

    @property
    def rest_rho(self):
        if self._rest_rho is not None:
            return self._rest_rho
        else:
            raise AttributeError("Must specify rest_rho for EnergyEvaluatorMix")

    @rest_rho.setter
    def rest_rho(self, value):
        self._rest_rho = value

    def __call__(self, rho, calcType=["E","V"], **kwargs):
        return self.compute(rho, calcType, **kwargs)

    def compute(self, rho, calcType=["E","V"], with_global = True):
        if self.sub_evaluator is None :
            potential = Field(grid=rho.grid, rank=1, direct=True)
            obj = Functional(name = 'ZERO', energy=0.0, potential=potential)
        else :
            obj = self.sub_evaluator(rho, calcType = calcType)
        if self.embed_evaluator is not None :
            obj -= self.embed_evaluator(rho, calcType = calcType)
        if with_global :
            self.gsystem.set_density(rho + self.rest_rho)
            obj_global = self.gsystem.total_evaluator(self.gsystem.density, calcType = calcType)
            if 'V' in calcType :
                if hasattr(obj, 'potential'):
                    obj.potential += self.gsystem.sub_value(obj_global.potential, rho)
                else :
                    obj.potential = self.gsystem.sub_value(obj_global.potential, rho)
            if 'E' in calcType :
                if hasattr(obj, 'energy'):
                    obj.energy += obj_global.energy
                else :
                    obj.energy = obj_global.energy
        return obj
