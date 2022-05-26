import numpy as np
from edftpy.utils.common import Field, Functional, AbsFunctional
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
        # calcType = ['E', 'V']
        eplist = {}
        results = None
        for key, evalfunctional in self.funcdicts.items():
            obj = evalfunctional(density, calcType = calcType, **kwargs)
            # if hasattr(obj, 'energy'): sprint(key, density.mp.asum(obj.energy), comm = density.mp.comm)
            # if hasattr(obj, 'energy'): sprint(key, density.mp.asum(obj.energy * 27.21138), comm = density.mp.comm)
            # if hasattr(obj, 'potential'): sprint(key, obj.potential[:3, 0, 0] * 2, comm = density.mp.comm)
            if results is None :
                results = obj
            else :
                results += obj
            if 'E' in calcType and split and gather :
                obj.energy = density.mp.vsum(obj.energy)
            eplist[key] = obj
        if results is None :
            results = Functional(name = 'ZERO')
            if 'E' in calcType :
                results.energy = 0.0
            if 'V' in calcType :
                results.potential = Field(grid=density.grid, rank=1, direct=True)

        if 'E' in calcType and gather :
            results.energy = density.mp.vsum(results.energy)
        #-----------------------------------------------------------------------
        if split :
            eplist['TOTAL'] = results
            return eplist
        #-----------------------------------------------------------------------
        return results

    def update_functional(self, remove = [], add = {}):
        removed = {}
        for key in remove :
            if key in self.funcdicts :
                removed[key] = self.funcdicts.pop(key)
        self.funcdicts.update(add)
        return removed

class EmbedEvaluator(Evaluator):
    def __init__(self, ke_evaluator = None, **kwargs):
        """
        """
        super().__init__(**kwargs)

        self._ke_evaluator = ke_evaluator
        self.embed_potential = None
        self.global_potential = None

    def get_embed_potential(self, rho, gaussian_density = None, with_ke = False, gather = False, with_global = True, **kwargs):
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

        if gather :
            self.embed_potential = self.embed_potential.gather()

        if with_global:
            if self.global_potential is not None and \
                    self.embed_potential.shape[-3:] == self.global_potential.shape[-3:] :
                self.embed_potential += self.global_potential

    @property
    def ke_evaluator(self):
        return self._ke_evaluator

    @ke_evaluator.setter
    def ke_evaluator(self, value):
        self._ke_evaluator = value

    def __call__(self, rho, calcType=["E","V"], with_ke = True, with_embed = False, **kwargs):
        return self.compute(rho, calcType, with_ke, with_embed, **kwargs)

    def compute(self, rho, calcType=["E","V"], with_ke = True, with_embed = False, gather = False, split = False, **kwargs):
        eplist = {}
        obj = None
        if with_embed :
            potential = self.embed_potential
            energy = np.sum(rho * self.embed_potential) * rho.grid.dV
            obj = Functional(name = 'Embed', energy=energy, potential=potential)
        elif self.nfunc > 0 :
            obj = super().compute(rho, calcType = calcType, gather = False, split = split, **kwargs)
            if split :
                eplist = obj
                obj = eplist.pop('TOTAL')
            obj *= -1.0

        if self.ke_evaluator is not None and with_ke:
            obj_ke = self.ke_evaluator(rho, calcType = calcType, **kwargs)
            if obj is None :
                obj = obj_ke
            else :
                obj += obj_ke

            if split : eplist['KEE'] = obj_ke

        if obj is None :
            obj = Functional(name = 'ZERO', energy=0.0)
            if 'V' in calcType :
                obj.potential = Field(grid=rho.grid, rank=1, direct=True)

        if 'E' in calcType and gather :
            obj.energy = rho.mp.vsum(obj.energy)

        if split :
            if gather :
                for key, item in eplist.items() :
                    eplist[key].energy = rho.mp.vsum(item.energy)
            eplist['TOTAL'] = obj
            return eplist
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
            obj = Functional(name = 'ZERO', energy=0.0)
            if 'V' in calcType :
                obj.potential = Field(grid=rho.grid, rank=1, direct=True)
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
        self.static_potential = None
        self.embed_energydensity= None
        self.external_potential = {}

    def get_embed_potential(self, rho, gaussian_density = None, embed_keys = [], with_global = True, calcType = ['V'], **kwargs):
        self.embed_potential = None
        if not embed_keys : embed_keys = self.embed_keys
        key = 'KE' if hasattr(self, 'KE') else None
        remove_global = {}
        if key is not None :
            func = getattr(self, key)
            remove_global = {key : func}
            if gaussian_density is not None :
                obj_global = func(rho + gaussian_density, calcType = calcType, **kwargs)
            else :
                obj_global = func(rho, calcType = calcType, **kwargs)
            self.embed_potential = obj_global.potential
            if 'D' in calcType :
                self.embed_energydensity = obj_global.energydensity
        #-----------------------------------------------------------------------
        if not with_global :
            for key in self.funcdicts:
                if key not in embed_keys :
                    remove_global[key] = self.funcdicts[key]
        self.update_functional(remove = remove_global)
        #-----------------------------------------------------------------------
        static_funcs = {'HARTREE' : None, 'PSEUDO' : None}
        remove_global2 = {}
        for key in self.funcdicts:
            if key not in static_funcs:
                remove_global2[key] = self.funcdicts[key]
            else :
                static_funcs[key] = self.funcdicts[key]
        self.update_functional(remove = remove_global2)
        obj_global = self.compute(rho, calcType = calcType, **kwargs)
        self.external_potential['QM'] = obj_global.potential.copy()
        if len(remove_global2) > 0 :
            self.update_functional(remove = static_funcs)
            self.update_functional(add = remove_global2)
            obj_global += self.compute(rho, calcType = calcType, **kwargs)
            self.update_functional(add = static_funcs)
        #-----------------------------------------------------------------------
        if self.embed_potential is None :
            self.embed_potential = obj_global.potential
        else :
            self.embed_potential += obj_global.potential
        self.update_functional(add = remove_global)
        if 'D' in calcType :
            if self.embed_energydensity is None :
                self.embed_energydensity = obj_global.energydensity
            else :
                self.embed_energydensity += obj_global.energydensity

        # self.static_potential = obj_global.potential

    def get_total_functional(self, rho, embed_keys = [], calcType = ['V'], **kwargs):
        remove_global = {}
        for key in self.funcdicts:
            if key in embed_keys:
                remove_global[key] = self.funcdicts[key]
        self.update_functional(remove = remove_global)
        obj = self.compute(rho, calcType = calcType, **kwargs)
        self.update_functional(add = remove_global)
        return obj

    def __call__(self, rho, calcType=["E","V"], **kwargs):
        return self.compute(rho, calcType, **kwargs)

    def compute(self, rho, calcType=["E","V"], gather = False, olevel = 0, **kwargs):
        if olevel == 0 :
            obj = super().compute(rho, calcType = calcType, gather = gather, **kwargs)
        else :
            obj = Functional(name = 'ZERO', potential = self.embed_potential)
            if 'E' in calcType :
                obj.energy = np.sum(rho * self.embed_potential) * rho.grid.dV
            if gather:
                obj.energy = rho.mp.vsum(obj.energy)

        return obj
