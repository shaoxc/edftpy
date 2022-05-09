import numpy as np

from edftpy.utils.math import r2s, s2r
from .force import *
from .potential import get_electrostatic_potential

def get_dipole(density, ions = None, rp = None):
    if rp is None :
        if ions is None :
            raise AttributeError("At least given ions")
        rp = get_charge_vector(density, ions)
    dip = np.zeros(3)
    grid = density.grid
    for i in range(len(dip)):
        dip[i] = grid.mp.einsum('ijk, ijk->', rp[i], density)*grid.dV
    return dip

def get_charge_vector(density, ions):
    grid = density.grid
    r0 = np.zeros(3)
    ncharge = 0.0
    for item, n in zip(*np.unique(ions.labels, return_counts=True)):
        ncharge += ions.Zval[item] * n
        r0 += np.sum(ions.pos.to_cart()[ions.labels == item], axis = 0)*ions.Zval[item]
    r0 /= ncharge
    rp = density.grid.r - np.expand_dims(r0, axis = (1, 2, 3))
    rp = np.moveaxis(rp, 0, -1)
    rp = r2s(rp, grid)
    rp -= np.rint(rp)
    rp = s2r(rp, grid)
    rp = np.moveaxis(rp, -1, 0)
    return rp

def get_total_energies(gsystem = None, drivers = None, density = None, total_energy= None, update = True, olevel = 0, **kwargs):
    elist = []
    if density is None :
        density = gsystem.density.copy()
    if total_energy is None :
        total_energy = gsystem.total_evaluator(density, calcType = ['E'], olevel = olevel).energy
    elist.append(total_energy)

    if isinstance(update, bool):
        update = [update,]*len(drivers)

    for i, driver in enumerate(drivers):
        if driver is None :
            ene = 0.0
        elif not update[i]:
            ene = driver.energy
        else :
            gsystem.density[:] = density
            driver(density =driver.density, gsystem = gsystem, calcType = ['E'], olevel = olevel, **kwargs)
            ene = driver.energy
        elist.append(ene)
    elist = np.asarray(elist)
    elist = gsystem.grid.mp.vsum(elist)
    return elist
