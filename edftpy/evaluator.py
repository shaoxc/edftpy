import numpy as np
from .utils.common import Field, Functional, AbsFunctional
from .kedf import KEDF
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
        # mask = density < 1E-30
        # saved = density[mask].copy()
        # print('density', density[density < 1E-30])
        # density[mask] = 1E-30
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
        #-----------------------------------------------------------------------
        # density[mask] = saved
        #-----------------------------------------------------------------------
        return results

    def update_functional(self, remove = [], add = {}):
        for key in remove :
            if key in self.funcdicts :
                del self.funcdicts[key]
        self.funcdicts.update(add)


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
        self.embed_potential = None

    def get_sub_evaluator(self, sub_functionals, **kwargs):
        funcdicts = {}
        for item in sub_functionals :
            func = FunctionalClass(type=item['type'], name = item['name'], **item['options'])
            funcdicts[item['type']] = func
        sub_evaluator = Evaluator(**funcdicts)
        return sub_evaluator

    def get_embed_potential(self, rho, core_density = None, **kwargs):
        if self.embed_evaluator is None :
            self.embed_potential = Field(grid=rho.grid, rank=1, direct=True)
            return
        #-----------------------------------------------------------------------
        if core_density is not None :
            # key = 'KE'
            for key, value in self.embed_evaluator.funcdicts.items():
                if isinstance(value, KEDF):
                    break
            remove_embed = {key : self.embed_evaluator.funcdicts[key]}
            ke_embed = self.embed_evaluator.funcdicts[key]
            self.embed_evaluator.update_functional(remove = remove_embed)
            for key, value in self.gsystem.total_evaluator.funcdicts.items():
                if isinstance(value, KEDF):
                    break
            remove_global = {key : self.gsystem.total_evaluator.funcdicts[key]}
            ke_global = self.gsystem.total_evaluator.funcdicts[key]
            self.gsystem.total_evaluator.update_functional(remove = remove_global)
            print('#-----------------------------------------------------------------------')
            print('core_density', np.sum(core_density), key)
            print('#-----------------------------------------------------------------------')
            self.embed_potential = -ke_embed(rho + core_density, calcType = ['V']).potential
            self.gsystem.set_density(rho + self.rest_rho + core_density)
            obj_global = ke_global(self.gsystem.density, calcType = ['V'])
            self.embed_potential = self.gsystem.sub_value(obj_global.potential, rho)
        else :
            remove_global = {}
            remove_embed = {}
            self.embed_potential = Field(rho.grid)
        #-----------------------------------------------------------------------
        self.gsystem.set_density(rho + self.rest_rho)
        self.embed_potential -= self.embed_evaluator(rho, calcType = ['V']).potential
        for key in self.gsystem.total_evaluator.funcdicts:
            if key not in self.embed_evaluator.funcdicts:
                remove_global[key] = self.gsystem.total_evaluator.funcdicts[key]
        self.gsystem.total_evaluator.update_functional(remove = remove_global)
        obj_global = self.gsystem.total_evaluator(self.gsystem.density, calcType = ['V'])
        self.embed_potential += self.gsystem.sub_value(obj_global.potential, rho)
        #-----------------------------------------------------------------------
        self.embed_evaluator.update_functional(add = remove_embed)
        self.gsystem.total_evaluator.update_functional(add = remove_global)
        #-----------------------------------------------------------------------

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
        if self.embed_evaluator is None :
            return self.compute_all(rho, calcType, **kwargs)
        else :
            return self.compute(rho, calcType, **kwargs)

    def compute(self, rho, calcType=["E","V"], with_global = True, **kwargs):
        if self.sub_evaluator is None :
            potential = Field(grid=rho.grid, rank=1, direct=True)
            obj = Functional(name = 'ZERO', energy=0.0, potential=potential)
        else :
            obj = self.sub_evaluator(rho, calcType = calcType)

        if 'V' in calcType :
            obj.potential += self.embed_potential
        if 'E' in calcType :
            obj.energy += np.sum(rho * self.embed_potential) * rho.grid.dV

        if with_global :
            #-----------------------------------------------------------------------
            remove_global = {}
            for key in self.embed_evaluator.funcdicts :
                remove_global[key] = self.gsystem.total_evaluator.funcdicts[key]
                self.gsystem.total_evaluator.update_functional(remove = remove_global)
            #-----------------------------------------------------------------------
            self.gsystem.set_density(rho + self.rest_rho)
            obj_global = self.gsystem.total_evaluator(self.gsystem.density, calcType = calcType)
            if 'V' in calcType :
                obj.potential += self.gsystem.sub_value(obj_global.potential, rho)
            if 'E' in calcType :
                obj.energy += obj_global.energy
            #-----------------------------------------------------------------------
            self.gsystem.total_evaluator.update_functional(add = remove_global)
            #-----------------------------------------------------------------------
        return obj

    def compute_all(self, rho, calcType=["E","V"], with_global = True, **kwargs):
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
                obj.potential += self.gsystem.sub_value(obj_global.potential, rho)
            if 'E' in calcType :
                obj.energy += obj_global.energy
        return obj
