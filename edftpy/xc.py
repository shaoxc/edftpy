from .utils.common import AbsFunctional

from dftpy.semilocal_xc import LibXC


class XC(AbsFunctional):
    def __init__(self, x_str='lda_x', c_str='lda_c_pz', core_density = None, **kwargs):
        self.options = {'x_str':x_str, 'c_str':c_str}
        self.options.update(kwargs)
        self._core_density = core_density

    @property
    def core_density(self):
        return self._core_density

    @core_density.setter
    def core_density(self, value):
        self._core_density = value

    def __call__(self, density, calcType=["E","V"], **kwargs):
        return self.compute(density, calcType=calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], **kwargs):
        self.options.update(kwargs)
        core_density = self.core_density
        if core_density is None :
            new_density = density
        elif density.rank == core_density.rank :
            new_density = density + core_density
        elif density.rank == 2 and core_density.rank == 1 :
            new_density = density.copy()
            new_density[0] += 0.5 * core_density
            new_density[1] += 0.5 * core_density

        # print('negative rho :', new_density[new_density < 0].sum() * density.grid.dV)
        xc = self.options.get('xc', None)
        if xc == 'PBE' :
            xc_kwargs = {"x_str":"gga_x_pbe", "c_str":"gga_c_pbe"}
        elif xc == 'LDA' :
            xc_kwargs = {"x_str":"lda_x", "c_str":"lda_c_pz"}
        else :
            xc_kwargs = {}
        self.options.update(xc_kwargs)
        functional = LibXC(density=new_density, calcType = calcType, **self.options)
        return functional
