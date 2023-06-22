import numpy as np
from edftpy.utils.math import fermi_dirac
from dftpy.density import DensityGenerator, file2density, gen_gaussian_density, build_pseudo_density, get_3d_value_recipe

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
