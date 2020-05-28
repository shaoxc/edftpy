from .utils.common import AbsFunctional

from dftpy.semilocal_xc import LibXC


class XC(AbsFunctional):
    def __init__(self, x_str='lda_x', c_str='lda_c_pz', core_density = None, **kwargs):
        self.options = {'x_str':x_str, 'c_str':c_str, 'core_density' :core_density}
        self.options.update(kwargs)

    def __call__(self, density, calcType=["E","V"], **kwargs):
        self.options.update(kwargs)
        return self.compute(density, calcType=calcType, **self.options)

    def compute(self, density, calcType=["E","V"], **kwargs):
        self.options.update(kwargs)
        if 'core_density' in self.options :
            core_density = self.options.pop('core_density')
            if core_density is None :
                new_density = density
            elif density.rank == core_density.rank :
                new_density = density + core_density
            elif density.rank == 2 and core_density.rank == 1 :
                new_density = density.copy()
                new_density[0] += 0.5 * core_density
                new_density[1] += 0.5 * core_density
        else :
            new_density = density
        functional = LibXC(density=new_density, calcType = calcType, **self.options)
        return functional
