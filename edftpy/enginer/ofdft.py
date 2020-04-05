from ..utils.common import AbsOptDriver

from dftpy.external_potential import ExternalPotential
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.optimization import Optimization
from dftpy.field import DirectField
from dftpy.grid import DirectGrid


# class EnergyEvaluatorMix(AbsFunctional):
class EnergyEvaluatorMix(object):

    def __init__(self, total_evaluator = None, embed_evaluator = None, sub_functionals = None, **kwargs):
        """
        sub_functionals : [dict{'type', 'name', 'options'}]
        """
        self.total_evaluator = total_evaluator
        self.embed_evaluator = embed_evaluator
        self.sub_evaluator = self.get_sub_evaluator(sub_functionals)
        self._gsystem = None

    def get_sub_evaluator(self, sub_functionals, **kwargs):
        funcdicts = {}
        for item in sub_functionals :
            func = FunctionalClass(type=item['type'], name = item['name'], **item['options'])
            funcdicts[item['type']] = func
        sub_evaluator = TotalEnergyAndPotential(**funcdicts)
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

    def compute(self, rho, calcType=["E","V"]):
        self.gsystem.set_density(rho + self.rest_rho)
        total_rho = self.gsystem.density
        obj = self.sub_evaluator(rho, calcType = calcType)
        obj -= self.embed_evaluator(rho, calcType = calcType)
        obj_global = self.total_evaluator(total_rho, calcType = calcType)
        if 'E' in calcType :
            obj.potential += self.gsystem.sub_value(obj_global.potential, rho)
        if 'V' in calcType :
            obj.energy += obj_global.energy
        return obj

class DFTpyOptDriver(AbsOptDriver):

    def __init__(self, total_evaluator = None, embed_evaluator = None, sub_functionals = None, options=None, grid = None, **kwargs):
        default_options = {
            "method" :'CG-HS',
            "maxcor": 6,
            "ftol": 1.0e-7,
            "xtol": 1.0e-12,
            "maxfun": 50,
            "maxiter": 200,
            "maxls": 30,
            "econv": 1.0e-3,
            "c1": 1e-4,
            "c2": 0.2,
            "algorithm": "EMM",
            "vector": "Orthogonalization",
        }
        self.options = default_options
        if options is not None :
            self.options.update(options)

        if total_evaluator is None:
            raise AttributeError("Must provide an total functionals evaluator")
        else:
            self.total_evaluator = total_evaluator

        if sub_functionals is None:
            raise AttributeError("Must provide an subsystem functionals")
        else:
            self.sub_functionals = sub_functionals
        self.grid = grid

        self.energy_evaluator = None
        self.get_energy_evaluator(total_evaluator, embed_evaluator, sub_functionals, **kwargs)

    def input_format(self, density = None, **kwargs):
        if isinstance(density, (DirectField)):
            self.rho = density
        else :
            if self.grid is None :
                grid = density.grid
                self.grid = DirectGrid(lattice=grid.lattice, nr=grid.nr, full=grid.full)
            self.rho = DirectField(grid=self.grid, griddata_3d = density, rank=1)
        return self.rho

    def get_energy_evaluator(self, total_evaluator = None, embed_evaluator = None, sub_functionals = None, **kwargs):
        """
        """
        if self.energy_evaluator is None :
            self.energy_evaluator = EnergyEvaluatorMix(
                    total_evaluator, embed_evaluator, sub_functionals, **kwargs)
        return self.energy_evaluator

    def get_sub_energy(self, density, calcType=["E","V"]):
        obj = self.energy_evaluator.sub_evaluator(density, calcType)
        obj -= self.energy_evaluator.embed_evaluator(density, calcType)
        return obj

    def __call__(self, density = None, gsystem = None, calcType = ['O', 'E', 'V'], ext_pot = None):
        return self.compute(density, gsystem, calcType, ext_pot)

    def compute(self, density = None, gsystem = None, calcType = ['O', 'E', 'V'], ext_pot = None):
        #-----------------------------------------------------------------------
        if density is None and self.rho is None:
            raise AttributeError("Must provide a guess density")
        elif density is not None :
            self.rho = density

        self.energy_evaluator.gsystem = gsystem
        rho_ini = self.input_format(self.rho)
        self.energy_evaluator.rest_rho = gsystem.sub_value(gsystem.density, rho_ini) - rho_ini
        #-----------------------------------------------------------------------
        if ext_pot is not None :
            ext = ExternalPotential(ext_pot)
            self.EnergyEvaluator.UpdateFunctional(newFuncDict={'EXT': ext})
        #-----------------------------------------------------------------------
        if 'O' in calcType :
            optimizer = Optimization(EnergyEvaluator=self.energy_evaluator, guess_rho=rho_ini, optimization_options=self.options)
            optimizer.optimize_rho()
            self.density = optimizer.rho
            # mu = (optimize.potential * self.density).integral() / self.density.N
            self.mu = optimizer.mu
        else :
            self.density = rho_ini

        if 'E' in calcType or 'V' in calcType :
            func = self.get_sub_energy(self.rho, calcType)
            self.functional = func

        if 'E' in calcType :
            self.energy = func.energy

        if 'V' in calcType :
            self.potential = func.potential
