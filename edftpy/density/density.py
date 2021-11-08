import numpy as np
import os
from edftpy import io
from dftpy.ewald import CBspline
from edftpy.utils.common import Field, RadialGrid
from edftpy.utils.math import fermi_dirac
from edftpy.mpi import sprint

def get_3d_value_recipe(r, arho, ions, grid, ncharge = None, rho = None, dtol=0.0, direct=True, pme=True, order=10, **kwargs):
    """
    """
    if hasattr(grid, 'get_reciprocal'):
        reciprocal_grid = grid.get_reciprocal()
    else :
        reciprocal_grid = grid
        grid = grid.get_direct()

    rho_g = Field(reciprocal_grid, direct = False)

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
            rho_g += vlines[key] * qarray.fft()
        rho_g *= Bspline.Barray * grid.nnrR / grid.volume
    else :
        for key in r:
            r0 = r[key]
            arho0 = arho[key]
            radial[key] = RadialGrid(r0, arho0, direct = False)
            vlines[key] = radial[key].to_3d_grid(reciprocal_grid.q)
            for i in range(ions.nat):
                if ions.labels[i] == key:
                    strf = ions.strf(reciprocal_grid, i)
                    rho_g += vlines[key] * strf

    if direct :
        rho = rho_g.ifft()
        rho[rho < dtol] = dtol
        sprint('Guess density (Real): ', rho.integral(), comm=grid.mp.comm)
        if ncharge is None :
            ncharge = 0.0
            for i in range(ions.nat) :
                ncharge += ions.Zval[ions.labels[i]]
        rho[:] *= ncharge / (rho.integral())
        sprint('Guess density (Scale): ', rho.integral(), comm=grid.mp.comm)
    return rho

def normalization_density(density, ncharge = None, grid = None, tol = 1E-300, method = 'shift'):
    if grid is None :
        grid = density.grid
    # min_rho = grid.mp.amin(density)
    # sprint('min0', min_rho, comm=grid.mp.comm)
    if method == 'scale' :
        if ncharge is not None :
            ncharge = np.sum(density) * grid.dV
        ncharge = grid.mp.asum(ncharge)
        density[density < tol] = tol
        nc = grid.mp.asum(np.sum(density) * grid.dV)
        density *= ncharge / nc
    elif method == 'shift' :
        if ncharge is not None :
            nc = grid.mp.asum(np.sum(density) * grid.dV)
            density += (ncharge-nc)/grid.nnrR
    else :
        pass
    if not hasattr(density, 'grid'):
        density = Field(grid, data=density, direct=True)
    # min_rho = grid.mp.amin(density)
    # sprint('min1', min_rho, comm=grid.mp.comm)
    return density

def filter_density(grid, density, mu = 0.7, kt = 0.05):
    grid._filter = 1.0
    # Not use _filter
    ncharge = np.sum(density)
    if grid._filter is None :
        recip_grid = density.grid.get_reciprocal()
        q = recip_grid.q
        mu *= np.max(q)
        kt *= mu
        grid._filter = fermi_dirac(q.real, mu = mu, kt = kt)
    den_g = density.fft() * grid._filter
    den = den_g.ifft(force_real=True)
    #-----------------------------------------------------------------------
    den = density
    #-----------------------------------------------------------------------
    den *= ncharge/np.sum(den)
    den[den < 0] = 1E-300
    return den

def file2density(filename, density, grid = None):
    if grid is None :
        grid = density.grid
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".snpy":
        density[:] = io.read_density(filename, grid=grid)
    else :
        fstr = f'!WARN : snpy format for density initialization will better, but this file is "{filename}".'
        sprint(fstr, comm=grid.mp.comm, level=2)
        if grid.mp.comm.rank == 0 :
            density = io.read_density(filename)
        else :
            density = np.zeros(1)
        grid.scatter(density, out = density)
