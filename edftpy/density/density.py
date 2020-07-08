import os
import numpy as np
from scipy.interpolate import splrep, splev
from numpy import linalg as LA
from edftpy.utils.common import Field, RadialGrid
from dftpy.pseudo import ReadPseudo
from dftpy.ewald import CBspline


def get_3d_value_recipe(r, arho, ions, grid, ncharge = None, rho = None, dtol=1E-30, direct=True, pme=True, order=10, **kwargs):
    """
    """
    if hasattr(grid, 'get_reciprocal'):
        reciprocal_grid = grid.get_reciprocal()
    else :
        reciprocal_grid = grid
        grid = grid.get_direct()

    rho = Field(reciprocal_grid, direct = False)

    radial = {}
    vlines = {}
    if pme :
        Bspline = CBspline(ions=ions, grid=grid, order=order)
        qa = np.empty(grid.nr)
        for key in r:
            r0 = r[key]
            arho0 = arho[key]
            radial[key] = RadialGrid(r0, arho0, direct = False)
            vlines[key] = radial[key].to_3d_grid(reciprocal_grid.q)
            qa[:] = 0.0
            for i in range(len(ions.pos)):
                if ions.labels[i] == key:
                    qa = Bspline.get_PME_Qarray(i, qa)
            qarray = Field(grid=grid, data=qa, direct = True)
            rho += vlines[key] * qarray.fft()
        rho *= Bspline.Barray * grid.nnr / grid.volume
    else :
        for key in r:
            r0 = r[key]
            arho0 = arho[key]
            radial[key] = RadialGrid(r0, arho0, direct = False)
            vlines[key] = radial[key].to_3d_grid(reciprocal_grid.q)
            for i in range(ions.nat):
                if ions.labels[i] == key:
                    strf = ions.strf(reciprocal_grid, i)
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

def normalization_density(density, ncharge = None, grid = None, tol = 1E-300):
    #-----------------------------------------------------------------------
    # minn = np.min(density) - tol
    # density -= minn
    #-----------------------------------------------------------------------
    print('min0', np.min(density))
    print('total0', np.sum(density)*grid.dV)
    # density[density < tol] = tol
    if ncharge is not None :
        if grid is None :
            grid = density.grid
        density += (ncharge-np.sum(density)*grid.dV)/np.size(density)
        density = Field(grid, data=density, direct=True)
    print('min1', np.min(density))
    print('total1', np.sum(density)*grid.dV)
    return density

def filter_density(self, density, mu = 0.7, kt = 0.05):
    self._filter = 1.0
    # Not use _filter
    ncharge = np.sum(density)
    if self._filter is None :
        recip_grid = density.grid.get_reciprocal()
        q = recip_grid.q
        mu *= np.max(q)
        kt *= mu
        self._filter = fermi_dirac(q.real, mu = mu, kt = kt)
    den_g = density.fft() * self._filter
    den = den_g.ifft(force_real=True)
    #-----------------------------------------------------------------------
    den = density
    #-----------------------------------------------------------------------
    den *= ncharge/np.sum(den)
    den[den < 0] = 1E-300
    return den
