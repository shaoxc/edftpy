from .utils.common import AbsFunctional

from dftpy.semilocal_xc import LibXC


class XC(AbsFunctional):
    def __init__(self, x_str='lda_x', c_str='lda_c_pz', **kwargs):
        self.options = {'x_str':x_str, 'c_str':c_str}
        self.options.update(kwargs)

    def __call__(self, density, calcType=["E","V"], **kwargs):
        self.options.update(kwargs)
        return self.compute(density, calcType=calcType, **self.options)

    def compute(self, density, calcType=["E","V"], **kwargs):
        self.options.update(kwargs)
        functional = LibXC(density=density, calcType = calcType, **kwargs)
        return functional
