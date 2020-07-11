import numpy as np
import time
import os

def get_forces(opt_drivers = None, gsystem = None, linearii=True):
    forces = gsystem.get_forces(linearii = linearii)
    for i, driver in enumerate(opt_drivers):
        fs = driver.calculator.get_forces()
        ind = driver.calculator.subcell.ions_index
        # print('ind', ind)
        # print('fs', fs)
        forces[ind] += fs
    return forces

def GetStress():
    pass
