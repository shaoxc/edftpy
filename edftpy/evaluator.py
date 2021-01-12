import numpy as np
from .utils.common import Field, Functional, AbsFunctional
from .utils.math import grid_map_index, grid_map_data
from .kedf import KEDF
from dftpy.functionals import FunctionalClass
from dftpy.formats.io import write
import scipy.special as sp

from edftpy.mpi import sprint


class Evaluator(AbsFunctional):
    def __init__(self, **kwargs):
        self.funcdicts = {}
        self.funcdicts.update(kwargs)
        remove = []
        for key, evalfunctional in self.funcdicts.items():
            if evalfunctional is None: remove.append(key)
        self.update_functional(remove = remove)

    @property
    def nfunc(self):
        return len(self.funcdicts)

    def __getattr__(self, attr):
        if attr in self.funcdicts :
            return self.funcdicts[attr]
        else :
            attr_u = attr.upper()
            if attr_u in self.funcdicts :
                return self.funcdicts[attr_u]
        raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def __call__(self, density, calcType=["E","V"], **kwargs):
        return self.compute(density, calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], split = False, gather = False, **kwargs):
        calcType = ['E', 'V']
        eplist = {}
        results = None
        for key, evalfunctional in self.funcdicts.items():
            obj = evalfunctional(density, calcType, **kwargs)
            # if hasattr(obj, 'energy'): sprint(key, density.mp.asum(obj.energy * 2), density.asum())
            # if hasattr(obj, 'energy'): sprint(key, density.mp.asum(obj.energy * 27.21138))
            # if hasattr(obj, 'potential'): sprint(key, obj.potential[:3, 0, 0] * 2)
            if results is None :
                results = obj
            else :
                results += obj
            eplist[key] = obj
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
        if split :
            eplist['TOTAL'] = results
            return eplist
        #-----------------------------------------------------------------------
        if 'E' in calcType and gather :
            results.energy = density.mp.vsum(results.energy)
        return results

    def update_functional(self, remove = [], add = {}):
        for key in remove :
            if key in self.funcdicts :
                del self.funcdicts[key]
        self.funcdicts.update(add)

class EmbedEvaluator(Evaluator):
    def __init__(self, ke_evaluator = None, **kwargs):
        """
        """
        super().__init__(**kwargs)

        self._ke_evaluator = ke_evaluator
        self.embed_potential = None
        self.global_potential = None

    def get_embed_potential(self, rho, gaussian_density = None, with_ke = False, gather = False, **kwargs):
        self.embed_potential = None
        key = 'KE' if hasattr(self, 'KE') else None
        remove_embed = {}
        if key is not None and gaussian_density is not None :
            ke_embed = getattr(self, key)
            remove_embed = {key : ke_embed}
            self.update_functional(remove = remove_embed)
            self.embed_potential = -ke_embed(gaussian_density + rho, calcType = ['V'], **kwargs).potential
        #-----------------------------------------------------------------------
        if self.embed_potential is None :
            self.embed_potential = self.compute(rho, calcType = ['V'], with_ke = with_ke, **kwargs).potential
        else :
            self.embed_potential += self.compute(rho, calcType = ['V'], with_ke = with_ke, **kwargs).potential

        self.update_functional(add = remove_embed)

        if self.global_potential is not None :
            if gather :
                self.embed_potential = self.embed_potential.gather()
                self.embed_potential += self.global_potential
            else :
                self.embed_potential += self.global_potential

    @property
    def ke_evaluator(self):
        return self._ke_evaluator

    @ke_evaluator.setter
    def ke_evaluator(self, value):
        self._ke_evaluator = value

    def __call__(self, rho, calcType=["E","V"], with_ke = True, with_embed = False, **kwargs):
        return self.compute(rho, calcType, with_ke, with_embed, **kwargs)

    def compute(self, rho, calcType=["E","V"], with_ke = True, with_embed = False, gather = False, **kwargs):
        if with_embed :
            potential = self.embed_potential
            energy = np.sum(rho * self.embed_potential) * rho.grid.dV
            obj = Functional(name = 'Embed', energy=energy, potential=potential)
        elif self.nfunc > 0 :
            obj = super().compute(rho, calcType = calcType, gather = False, **kwargs)
            obj *= -1.0
        else :
            if 'V' in calcType :
                potential = Field(grid=rho.grid, rank=1, direct=True)
            else :
                potential = None
            obj = Functional(name = 'ZERO', energy=0.0, potential=potential)

        if self.ke_evaluator is not None and with_ke:
            obj += self.ke_evaluator(rho, calcType = calcType, **kwargs)

        if 'E' in calcType and gather :
            obj.energy = rho.mp.vsum(obj.energy)
        return obj

class EvaluatorOF(Evaluator):
    def __init__(self, ke_evaluator = None, gsystem = None, **kwargs):
        """
        """
        super().__init__(**kwargs)

        self._gsystem = gsystem
        self._ke_evaluator = ke_evaluator

    @property
    def gsystem(self):
        return self._gsystem

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

    def set_rest_rho(self, subrho):
        self.rest_rho = - subrho
        self.gsystem.add_to_sub(self.gsystem.density, self._rest_rho)

    def __call__(self, rho, calcType=["E","V"], **kwargs):
        return self.compute(rho, calcType, **kwargs)

    def compute(self, rho, calcType=["E","V"], with_global = True, with_ke = False, with_sub = True, with_embed = True, gather = False, **kwargs):
        if self.nfunc == 0 or not with_sub :
            potential = Field(grid=rho.grid, rank=1, direct=True)
            obj = Functional(name = 'ZERO', energy=0.0, potential=potential)
        else :
            obj = super().compute(rho, calcType = calcType, gather = False, **kwargs)
        #Embedding potential
        if with_embed :
            if 'V' in calcType :
                obj.potential += self.embed_potential
            if 'E' in calcType :
                energy = np.sum(rho * self.embed_potential) * rho.grid.dV
                obj.energy += energy

        if self.ke_evaluator is not None and with_ke:
            obj += self.ke_evaluator(rho, calcType = calcType, **kwargs)

        if with_global :
            self.gsystem.set_density(rho + self.rest_rho)
            obj_global = self.gsystem.total_evaluator(self.gsystem.density, calcType = calcType, **kwargs)
            if 'V' in calcType :
                self.gsystem.add_to_sub(obj_global.potential, obj.potential)
            if 'E' in calcType :
                obj.energy += obj_global.energy
        if 'E' in calcType and gather:
            obj.energy = rho.mp.vsum(obj.energy)
        return obj

class TotalEvaluator(Evaluator):
    def __init__(self, embed_keys = [], **kwargs):
        super().__init__(**kwargs)
        self.embed_keys = embed_keys
        self.embed_potential = None
        self.local_potential = None
        self.static_potential = None

    def get_embed_potential(self, rho, gaussian_density = None, embed_keys = [], with_global = True, **kwargs):
        self.embed_potential = None
        if embed_keys :
            embed_keys = self.embed_keys
        key = 'KE' if hasattr(self, 'KE') else None
        remove_global = {}
        if key is not None :
            func = getattr(self, key)
            remove_global = {key : func}
            if gaussian_density is not None :
                obj_global = func(rho + gaussian_density, calcType = ['V'], **kwargs)
            else :
                obj_global = func(rho, calcType = ['V'], **kwargs)
            self.embed_potential = obj_global.potential
        #-----------------------------------------------------------------------
        if not with_global :
            for key in self.funcdicts:
                if key not in embed_keys :
                    remove_global[key] = self.funcdicts[key]
        self.update_functional(remove = remove_global)
        obj_global = self.compute(rho, calcType = ['V'], **kwargs)
        if self.embed_potential is None :
            self.embed_potential = obj_global.potential
        else :
            self.embed_potential += obj_global.potential
        self.update_functional(add = remove_global)

        self.static_potential = obj_global.potential
