from edftpy.functional import LocalPP, KEDF, Hartree, XC, Ewald

class PerturbationEnergy :
    def __init__(self, evaluator, **kwargs):
        self.evaluator = evaluator

    def __call__(self, rho, rhod, order = 1, **kwargs):
        if order == 1 :
            energy = self.energy_1(rho, rhod, **kwargs)
        elif order == 2 :
            energy = self.energy_2(rhod, rhod, **kwargs)
        else :
            raise ValueError('Only support up to second order')
        return energy

    def energy_1(self, rho, rhod, **kwargs):
        obj = self.evaluator(rho, calcType = ('V'))
        if hasattr(obj, 'potential') :
            potential = self.evaluator(rho, calcType = ('V')).potential
            energy = (potential*rhod).sum()*rho.grid.dV
        else :
            energy = 0.0
        return energy

    def energy_2(self, rho, rhod, **kwargs):
        energy = 0.0
        kernel = 0.0
        for key, evalfunctional in self.evaluator.funcdicts.items():
            if isinstance(evalfunctional, Hartree):
                energy += evalfunctional(rhod, calcType = ('E')).energy*2.0
            else :
                obj = evalfunctional(rho, calcType = ('V2'), **kwargs)
                if hasattr(obj, 'v2rho2') :
                    kernel = kernel + obj.v2rho2
        energy += (kernel*rhod**2).sum()* rhod.grid.dV
        energy *= 0.5
        return energy
