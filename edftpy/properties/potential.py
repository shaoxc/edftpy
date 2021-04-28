import numpy as np

def get_electrostatic_potential(gsystem, keys =['PSEUDO', 'HARTREE']):
    # vloc = gsystem.total_evaluator.funcdicts['PSEUDO']
    # vhart = gsystem.total_evaluator.funcdicts['HARTREE']
    # v = vloc(gsystem.density, calcType = ['V']).potential + vhart(gsystem.density, calcType = ['V']).potential
    keys_wf = keys
    keys_global = gsystem.total_evaluator.funcdicts.keys()
    remove_global = {key : gsystem.total_evaluator.funcdicts[key] for key in keys_global if key not in keys_wf}
    gsystem.total_evaluator.update_functional(remove = remove_global)
    wf_pot = gsystem.total_evaluator(gsystem.density, calcType = ['V']).potential
    gsystem.total_evaluator.update_functional(add = remove_global)

    # wf_pot_z = np.sum(wf_pot, axis = (0, 1)) * ENERGY_CONV['Hartree']['eV']/np.prod(wf_pot.shape[:2])
    # zpos = np.linspace(0, length, wf_pot_z.size)
    return wf_pot
