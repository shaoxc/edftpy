import numpy as np
from dftpy.constants import ENERGY_CONV, LEN_CONV

from .utils.common import AbsFunctional
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


def electric_potential(evaluator, density, length = 1.0):
    keys_wf = ['PSEUDO', 'HARTREE']
    keys_global = evaluator.funcdicts.keys()
    remove_global = {key : evaluator.funcdicts[key] for key in keys_global if key not in keys_wf}
    evaluator.update_functional(remove = remove_global)

    wf_pot = evaluator(density, calcType = ['V']).potential
    wf_pot_z = np.sum(wf_pot, axis = (0, 1)) * ENERGY_CONV['Hartree']['eV']/np.prod(wf_pot.shape[:2])
    zpos = np.linspace(0, length, wf_pot_z.size)

    evaluator.update_functional(add = remove_global)
    return (zpos, wf_pot_z)
