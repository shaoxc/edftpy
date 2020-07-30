import numpy as np
from .utils.common import Field, Functional, AbsFunctional
from .utils.math import grid_map_index, grid_map_data
from .kedf import KEDF
from dftpy.functionals import FunctionalClass
from dftpy.formats.io import write
import scipy.special as sp


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
        for key, evalfunctional in self.funcdicts.items():
            obj = evalfunctional(density, calcType)
            # if hasattr(obj, 'energy'): print(key, obj.energy * 27.21138)
            # if hasattr(obj, 'potential'): print(key, obj.potential[:3, 0, 0] * 2)
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

    def update_functional(self, remove = [], add = {}):
        for key in remove :
            if key in self.funcdicts :
                del self.funcdicts[key]
        self.funcdicts.update(add)

class EnergyEvaluatorMix(AbsFunctional):
    def __init__(self, embed_evaluator = None, ke_evaluator = None, mod_type = 1, gsystem = None, **kwargs):
        """
        sub_functionals : [dict{'type', 'name', 'options'}]
        """
        self.embed_evaluator = embed_evaluator
        self._gsystem = gsystem
        self.embed_potential = None
        self.mod_type = mod_type
        self._ke_evaluator = ke_evaluator
        self._iter = 0

    def get_embed_potential(self, rho, gaussian_density = None, with_global = False, with_ke = False, **kwargs):
        self._iter += 1
        print('gaussian_density', gaussian_density is not None, self.gsystem.gaussian_density is not None, self.mod_type)
        self.gsystem.set_density(rho + self.rest_rho)

        key = None
        if self.embed_evaluator is not None :
            for k, value in self.embed_evaluator.funcdicts.items():
                if isinstance(value, KEDF):
                    key = k
                    break
        # print('key', key, self.embed_evaluator.funcdicts.keys())
        if key is not None and gaussian_density is not None :
            remove_embed = {key : self.embed_evaluator.funcdicts[key]}
            ke_embed = self.embed_evaluator.funcdicts[key]
            self.embed_evaluator.update_functional(remove = remove_embed)
            for key, value in self.gsystem.total_evaluator.funcdicts.items():
                if isinstance(value, KEDF):
                    break
            remove_global = {key : self.gsystem.total_evaluator.funcdicts[key]}
            ke_global = self.gsystem.total_evaluator.funcdicts[key]
            #-----------------------------------------------------------------------
            if self.mod_type == 0 : # Without gaussian_density
                self.embed_potential = -ke_embed(rho, calcType = ['V']).potential
                obj_global = ke_global(self.gsystem.density, calcType = ['V'])
            elif self.mod_type == 1 : # gaussian_density add to subsytem and global system
                self.embed_potential = -ke_embed(gaussian_density + rho, calcType = ['V']).potential
                obj_global = ke_global(self.gsystem.gaussian_density + self.gsystem.density, calcType = ['V'])
            elif self.mod_type == 2 : # gaussian_density only add to global system (the worst)
                self.embed_potential = -ke_embed(rho, calcType = ['V']).potential
                obj_global = ke_global(self.gsystem.gaussian_density + self.gsystem.density, calcType = ['V'])
            elif self.mod_type == 3 : # Only use gaussian_density
                self.embed_potential = -ke_embed(gaussian_density, calcType = ['V']).potential
                obj_global = ke_global(self.gsystem.gaussian_density, calcType = ['V'])
            self.embed_potential += self.gsystem.sub_value(obj_global.potential, rho)
            #-----------------------------------------------------------------------
            print('pot2', np.min(self.embed_potential), np.max(self.embed_potential))
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

        if with_ke and self.ke_evaluator is not None :
            if self._iter < 0 : # debug most stable
                pot = self.ke_evaluator(rho, calcType = ['V'], ldw = 0.1).potential
            else :
                pot = self.ke_evaluator(rho, calcType = ['V']).potential
            self.embed_potential += pot

        print('pot3', np.min(self.embed_potential), np.max(self.embed_potential))

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
        # if self._ke_evaluator is not None:
            # return self._ke_evaluator
        # else:
            # raise AttributeError("Must specify ke_evaluator for EnergyEvaluatorMix")
        return self._ke_evaluator

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

    def __call__(self, rho, calcType=["E","V"], with_global = True, with_ke = True, with_embed = False, **kwargs):
        return self.compute(rho, calcType, with_global, with_ke, with_embed, **kwargs)

    def compute(self, rho, calcType=["E","V"], with_global = True, with_ke = True, with_embed = False, **kwargs):
        if with_embed :
            potential = self.embed_potential
            energy = np.sum(rho * self.embed_potential) * rho.grid.dV
            obj = Functional(name = 'Embed', energy=energy, potential=potential)
        elif self.embed_evaluator is not None :
            obj = self.embed_evaluator(rho, calcType = calcType)
            obj *= -1.0
        else :
            potential = Field(grid=rho.grid, rank=1, direct=True)
            obj = Functional(name = 'ZERO', energy=0.0, potential=potential)

        if with_global :
            self.gsystem.set_density(rho + self.rest_rho)
            obj_global = self.gsystem.total_evaluator(self.gsystem.density, calcType = calcType)
            if 'V' in calcType :
                obj.potential += self.gsystem.sub_value(obj_global.potential, rho)
            if 'E' in calcType :
                obj.energy += obj_global.energy

        if self.ke_evaluator is not None and with_ke:
            # here we return exact NL-KEDF energy
            obj += self.ke_evaluator(rho, calcType = calcType, ldw = None)
            # obj += self.ke_evaluator(rho, calcType = calcType)
        return obj

class EvaluatorOF(AbsFunctional):
    def __init__(self, sub_evaluator = None, ke_evaluator = None, gsystem = None, **kwargs):
        """
        """
        self.sub_evaluator = sub_evaluator
        self._gsystem = gsystem
        self._ke_evaluator = ke_evaluator

    @property
    def gsystem(self):
        return self._gsystem
        # if self._gsystem is not None:
            # return self._gsystem
        # else:
            # raise AttributeError("Must specify gsystem for EvaluatorOF")

    @gsystem.setter
    def gsystem(self, value):
        self._gsystem = value

    @property
    def ke_evaluator(self):
        return self._ke_evaluator

    @ke_evaluator.setter
    def ke_evaluator(self, value):
        self._ke_evaluator = value

    @property
    def rest_rho(self):
        if self._rest_rho is not None:
            return self._rest_rho
        else:
            raise AttributeError("Must specify rest_rho for EvaluatorOF")

    @rest_rho.setter
    def rest_rho(self, value):
        self._rest_rho = value

    @property
    def embed_potential(self):
        if self._embed_potential is not None:
            return self._embed_potential
        else:
            raise AttributeError("Must specify embed_potential for EvaluatorOF")

    @embed_potential.setter
    def embed_potential(self, value):
        self._embed_potential = value

    def __call__(self, rho, calcType=["E","V"], with_global = True, with_ke = False, with_sub = True, with_embed = True, **kwargs):
        return self.compute(rho, calcType, with_global, with_ke, with_sub, with_embed, **kwargs)

    def compute(self, rho, calcType=["E","V"], with_global = True, with_ke = False, with_sub = True, with_embed = True, **kwargs):
        if self.sub_evaluator is None or not with_sub :
            potential = Field(grid=rho.grid, rank=1, direct=True)
            obj = Functional(name = 'ZERO', energy=0.0, potential=potential)
        else :
            obj = self.sub_evaluator(rho, calcType = calcType)
        #-----------------------------------------------------------------------
        # if not np.all(rho.grid.nr == self.embed_potential.grid.nr) :
            # self.embed_potential = grid_map_data(self.embed_potential, grid = rho.grid)
        # if not np.all(rho.grid.nr == self.rest_rho.grid.nr) :
            # self.rest_rho = grid_map_data(self.rest_rho, grid = rho.grid)
        #-----------------------------------------------------------------------
        #Embedding potential
        if with_embed :
            if 'V' in calcType :
                obj.potential += self.embed_potential
            if 'E' in calcType :
                energy = np.sum(rho * self.embed_potential) * rho.grid.dV
                obj.energy += energy

        if self.ke_evaluator is not None and with_ke:
            obj += self.ke_evaluator(rho, calcType = calcType)

        if with_global :
            self.gsystem.set_density(rho + self.rest_rho)
            obj_global = self.gsystem.total_evaluator(self.gsystem.density, calcType = calcType)
            if 'V' in calcType :
                obj.potential += self.gsystem.sub_value(obj_global.potential, rho)
            if 'E' in calcType :
                obj.energy += obj_global.energy
        return obj
