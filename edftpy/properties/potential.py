import numpy as np

def get_electrostatic_potential(gsystem):
    vloc = gsystem.total_evaluator.funcdicts['PSEUDO']
    vhart = gsystem.total_evaluator.funcdicts['HARTREE']
    v = vloc(gsystem.density, calcType = ['V']).potential + vhart(gsystem.density, calcType = ['V']).potential
    return v
