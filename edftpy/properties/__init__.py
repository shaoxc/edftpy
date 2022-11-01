import numpy as np

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
    r0 = np.zeros(3)
    ncharge = ions.get_ncharges()
    for item in set(ions.symbols):
        r0 += np.sum(ions.positions[ions.symbols == item], axis = 0)*ions.zval[item]
    r0 /= ncharge
    rp = density.grid.r - np.expand_dims(r0, axis = (1, 2, 3))
    rp = np.moveaxis(rp, 0, -1)
    # rp = r2s(rp, grid)
    rp = ions.cell.scaled_positions(rp)
    rp -= np.rint(rp)
    rp = ions.cell.cartesian_positions(rp)
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
