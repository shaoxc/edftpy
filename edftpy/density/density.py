import os
import numpy as np
from scipy.interpolate import splrep, splev
from numpy import linalg as LA
from edftpy.utils.common import Field, RadialGrid
from dftpy.pseudo import ReadPseudo

def get_3d_value_recipe(r, arho, ions, grid, ncharge = None, rho = None, dtol=1E-30, direct=True, **kwargs):
    """
    """
    rho = Field(grid, direct = False)

    radial = {}
    vlines = {}
    for key in r:
        r0 = r[key]
        arho0 = arho[key]
        radial[key] = RadialGrid(r0, arho0, direct = False)
        vlines[key] = radial[key].to_3d_grid(grid.q)
        for i in range(ions.nat):
            if ions.labels[i] == key:
                strf = ions.strf(grid, i)
                rho += vlines[key] * strf

    if direct :
        rho = rho.ifft()
        rho[rho < dtol] = dtol
        print('Guess density (Real): ', rho.integral())
        if ncharge is None :
            ncharge = 0.0
            for i in range(ions.nat) :
                ncharge += ions.Zval[ions.labels[i]]
        rho[:] *= ncharge / (rho.integral())
        print('Guess density (Scale): ', rho.integral())
    return rho

def get_3d_value_real(r, arho, ions, grid, ncharge = None, rho = None, dtol=1E-30, direct=True, **kwargs):
    """
     !!! Not test.
    """
    maxp = 15000
    gmax = 30

    rho = Field(grid, direct = False)
    radial = {}
    vlines = {}
    for key in r:
        r0, arho0 = ReadPseudo._real2recip(r[key], arho[key], 0.0, maxp, gmax)
        arho0 = arho[key]
        radial[key] = RadialGrid(r0, arho0, direct = False)
        vlines[key] = radial[key].to_3d_grid(grid.q)
        for i in range(ions.nat):
            if ions.labels[i] == key:
                strf = ions.strf(grid, i)
                rho += vlines[key] * strf

    if direct :
        rho = rho.ifft()
        rho[rho < dtol] = dtol
        print('Guess density (Real): ', rho.integral())
        if ncharge is None :
            ncharge = 0.0
            for i in range(ions.nat) :
                ncharge += ions.Zval[ions.labels[i]]
        rho[:] *= ncharge / (rho.integral())
        print('Guess density (Scale): ', rho.integral())
    return rho
