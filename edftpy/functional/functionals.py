from edftpy.utils.common import AbsFunctional
from .kedf import KEDF


class NonAdditiveKE(AbsFunctional):
    def __init__(self, nbnd = 4, nspin = 1, **kwargs):
        # self.options = {'name' :'GGA', 'k_str' : 'REVAPBEK'}
        # self.options = {'name' :'GGA', 'k_str' : 'LKT', 'gga_remove_vw' : True}
        self.options = {'name' :'TF', 'k_str' : 'REVAPBEK'}
        self.options.update(kwargs)
        self.ke_total = KEDF(**self.options)
        self.ke_sub= KEDF(**self.options)
        self.nbnd = nbnd
        self.nspin = nspin

    def __call__(self, density, calcType=["E","V"], **kwargs):
        return self.compute(density, calcType=calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], **kwargs):
        obj = self.ke_total(density, calcType = calcType)

        fac = density.integral()/self.nbnd
        if self.nspin == 1 :
            fac /= 2.0

        obj_sub = self.ke_sub(density/fac, calcType = calcType)
        obj = obj - obj_sub * fac

        # if 'E' in calcType :
            # print('NonAdditiveKE :', obj.energy, fac, self.nbnd, density.integral())
        return obj
