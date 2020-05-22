import numpy as np
from .utils.common import Field, Functional, AbsFunctional
from .kedf import KEDF
from dftpy.functionals import FunctionalClass
from dftpy.formats.io import write


class Evaluator(AbsFunctional):
    def __init__(self, **kwargs):
        self.funcdicts = {}
        self.funcdicts.update(kwargs)
        for key, evalfunctional in self.funcdicts.items():
            if evalfunctional is None:
                del self.funcdicts[key]
        # if Evaluator is empty, return None
        if len(self.funcdicts) == 0 :
            return None

    def __call__(self, density, calcType=["E","V"], **kwargs):
        return self.compute(density, calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], **kwargs):
        results = None
        #-----------------------------------------------------------------------
        # nc = np.sum(density)
        # density[density < 1E-30] = 1E-30
        # density *= nc/np.sum(density)
        #-----------------------------------------------------------------------
        # mask = density < 1E-30
        # saved = density[mask].copy()
        # print('density', density[density < 1E-30])
        # density[mask] = 1E-30
        #-----------------------------------------------------------------------
        for key, evalfunctional in self.funcdicts.items():
            obj = evalfunctional(density, calcType)
            # if hasattr(obj, 'energy'): print(key, obj.energy * 27.21138)
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
    def __init__(self, embed_evaluator = None, sub_evaluator = None, sub_functionals = None, ke_evaluator = None, **kwargs):
        """
        sub_functionals : [dict{'type', 'name', 'options'}]
        """
        self.embed_evaluator = embed_evaluator
        self.sub_evaluator = sub_evaluator
        # self.sub_evaluator = self.get_sub_evaluator(sub_functionals)
        self._gsystem = None
        self.embed_potential = None
        self._ke_evaluator = ke_evaluator

    def get_sub_evaluator(self, sub_functionals, **kwargs):
        funcdicts = {}
        for item in sub_functionals :
            func = FunctionalClass(type=item['type'], name = item['name'], **item['options'])
            funcdicts[item['type']] = func
        sub_evaluator = Evaluator(**funcdicts)
        return sub_evaluator

    def get_embed_potential(self, rho, core_density = None, with_global = False, with_all = True, mod_type = 2, **kwargs):
        self.gsystem.set_density(rho + self.rest_rho)
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
            #-----------------------------------------------------------------------
            if mod_type == 0 : # Without core_density
                self.embed_potential = -ke_embed(rho, calcType = ['V']).potential
                obj_global = ke_global(self.gsystem.density, calcType = ['V'])
            elif mod_type == 1 : # core_density add to subsytem and global system
                self.embed_potential = -ke_embed(core_density + rho, calcType = ['V']).potential
                obj_global = ke_global(self.gsystem.fake_core_density + self.gsystem.density, calcType = ['V'])
            elif mod_type == 2 : # core_density only add to global system
                self.embed_potential = -ke_embed(rho, calcType = ['V']).potential
                obj_global = ke_global(self.gsystem.fake_core_density + self.gsystem.density, calcType = ['V'])
            elif mod_type == 3 : # Only use core_density
                self.embed_potential = -ke_embed(core_density, calcType = ['V']).potential
                obj_global = ke_global(self.gsystem.fake_core_density, calcType = ['V'])
            elif mod_type == 101 : # Just for debug
                self.embed_potential = -ke_embed(rho, calcType = ['V']).potential
                self.gsystem.set_density(rho + self.rest_rho + core_density)
                # self.gsystem.set_density(core_density)
                obj_global = ke_global(self.gsystem.density, calcType = ['V'])
            #-----------------------------------------------------------------------
            self.embed_potential += self.gsystem.sub_value(obj_global.potential, rho)
        else :
            remove_global = {}
            remove_embed = {}
            self.embed_potential = Field(rho.grid)
        #-----------------------------------------------------------------------
        if self.embed_evaluator is not None :
            self.embed_potential -= self.embed_evaluator(rho, calcType = ['V']).potential
            self.embed_evaluator.update_functional(add = remove_embed)
            if not with_global :
                for key in self.gsystem.total_evaluator.funcdicts:
                    if key not in self.embed_evaluator.funcdicts:
                        remove_global[key] = self.gsystem.total_evaluator.funcdicts[key]
        self.gsystem.total_evaluator.update_functional(remove = remove_global)
        obj_global = self.gsystem.total_evaluator(self.gsystem.density, calcType = ['V'])
        self.embed_potential += self.gsystem.sub_value(obj_global.potential, rho)
        self.gsystem.total_evaluator.update_functional(add = remove_global)
        #-----------------------------------------------------------------------
        if with_all and self.sub_evaluator is not None :
            self.embed_potential += self.sub_evaluator(rho, calcType = ['V']).potential

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
    def ke_evaluator(self):
        if self._ke_evaluator is not None:
            return self._ke_evaluator
        else:
            raise AttributeError("Must specify ke_evaluator for EnergyEvaluatorMix")

    @ke_evaluator.setter
    def ke_evaluator(self, value):
        self._ke_evaluator = value

    @property
    def rest_rho(self):
        if self._rest_rho is not None:
            return self._rest_rho
        else:
            raise AttributeError("Must specify rest_rho for EnergyEvaluatorMix")

    @rest_rho.setter
    def rest_rho(self, value):
        self._rest_rho = value

    def __call__(self, rho, calcType=["E","V"], with_global = True, embed = True, only_ke = False, **kwargs):

        if only_ke :
            return self.compute_only_ke(rho, calcType, embed = embed, **kwargs)
        elif embed :
            return self.compute(rho, calcType, with_global = with_global, **kwargs)
        else :
            return self.compute_normal(rho, calcType, with_global = with_global, **kwargs)

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
            if self.embed_evaluator is not None :
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

    def compute_normal(self, rho, calcType=["E","V"], with_global = True, **kwargs):
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

    def compute_only_ke(self, rho, calcType=["E","V"], embed = True, **kwargs):
        obj = self.ke_evaluator(rho)
        # print('embed', embed, obj.energy)
        if embed :
            if 'V' in calcType :
                obj.potential += self.embed_potential
            if 'E' in calcType :
                obj.energy += np.sum(rho * self.embed_potential) * rho.grid.dV
        else :
            if self.sub_evaluator is not None :
                obj += self.sub_evaluator(rho, calcType = calcType)
            if self.embed_evaluator is not None :
                obj -= self.embed_evaluator(rho, calcType = calcType)
        # print('embed1', embed, obj.energy)
        return obj
