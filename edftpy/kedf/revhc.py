import numpy as np
import scipy.special as sp
from scipy import stats

from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV, ZERO
from dftpy.formats.io import write

from edftpy.mpi import graphtopo, sprint, pmi

def get_hc_params(density, tol = 0.01):

    rhog = np.abs(np.real(density.fft()))
    q = density.grid.get_reciprocal().q
    mp = density.mp
    if mp.size > 1 :
        raise AttributeError("Only works on Serial version")
    #-----------------------------------------------------------------------
    nc = density.integral()
    rhog /= nc
    rhog[rhog < 0.01] = 0
    mask = (q > 1.0-tol) & (q < 1.5+tol)
    rhogm = rhog[mask]
    nbins = int((0.5+2*tol)/0.0005)
    bins =stats.binned_statistic(q[mask], rhogm, 'sum', bins=nbins, range=(1.0-tol, 1.5+tol))
    values = bins.statistic
    inds = (bins.bin_edges[:-1] + bins.bin_edges[1:])*0.5
    if np.max(values) < ZERO :
        width = 1.5
    else :
        width = inds[np.argmax(values)]
    width = min(1.5, width)
    #-----------------------------------------------------------------------
    # inds = np.argsort(rhogm)
    # widths = q[mask][inds]
    # maxn = rhogm[inds]
    # mask2 = maxn > ZERO
    # n = np.count_nonzero(mask2)
    # if n > 0 :
    #     width = np.sum(widths[mask2])/n
    # else :
    #     width = 1.5
    #-----------------------------------------------------------------------
    # if mp.rank == 0 :
    #     rhog[0, 0, 0] = 0
    # maxn = rhog.max()
    # maxn_w = mp.amax(maxn)
    # if abs(maxn - maxn_w) < 1E-30 :
    #     index = np.argmax(rhog)
    #     width = q.ravel()[index]
    # else :
    #     width = 1.0
    # width = mp.amax(width)
    #-----------------------------------------------------------------------
    x = (width - 1.0)*4.0-2.0
    scale = sp.erfc(x) - 1
    #-----------------------------------------------------------------------
    print('width, scale :', width, scale)
    return scale

def update_kedf_params(optimizer, params):
    for i, driver in enumerate(optimizer.drivers):
        if driver is None : continue
        if driver.evaluator.ke_evaluator is not None :
            driver.evaluator.ke_evaluator.kwargs['params'] = params
            if hasattr(driver.mixer, 'restart'):
                driver.mixer.restart()
    #-----------------------------------------------------------------------
    return optimizer

def optimizer_hc_params(optimizer, params = [0.1, 0.46]):
    # return optimizer
    # scale = get_hc_params(optimizer.gsystem.density)
    # scale = get_hc_params_den(optimizer.gsystem.density)
    scale = get_hc_params_den_3(optimizer.gsystem.density)
    sprint('Update HC scale: ', scale)
    if len(params) > 1 :
        params[1] = scale
    else :
        params[0] = scale
    update_kedf_params(optimizer, params)
    sprint('Update HC params : ', params)
    optimizer.optimize()
    #-----------------------------------------------------------------------
    energy = optimizer.energy
    sprint('Final energy (a.u.)', energy)
    sprint('Final energy (eV)', energy * ENERGY_CONV['Hartree']['eV'])
    sprint('Final energy (eV/atom)', energy * ENERGY_CONV['Hartree']['eV']/optimizer.gsystem.ions.nat)
    return optimizer

def get_hc_params_den(density):
    mp = density.mp
    if mp.size > 1 :
        raise AttributeError("Only works on Serial version")
    #-----------------------------------------------------------------------
    nc = density.asum()
    # rho0 = density.mean()
    rho0 = np.median(density)
    rho_h = density[density > 1.5*rho0]
    width = np.sum(rho_h) / nc * 0.75
    scale = float(width)
    #-----------------------------------------------------------------------
    # x = (width - 1.0)*4.0-2.0
    # scale = sp.erfc(x) - 1
    #-----------------------------------------------------------------------
    # scale = min(1.0, width)
    # print('width, scale :', width, scale)
    return scale

def get_hc_params_den_2(density):
    from dftpy.kedf.gga import get_gga_p
    mp = density.mp
    if mp.size > 1 :
        raise AttributeError("Only works on Serial version")
    #-----------------------------------------------------------------------
    p = get_gga_p(density)
    p = np.abs(p)
    p0 = np.max(p)
    x = (p - p0*0.5)/p0*4
    scale = (sp.erf(x) + 1)*0.5
    return scale

def get_hc_params_den_3(density):
    mp = density.mp
    if mp.size > 1 :
        raise AttributeError("Only works on Serial version")
    #-----------------------------------------------------------------------
    nc = density.asum()
    rho0 = np.median(density)
    rho_h = density[density > 1.25*rho0]
    width = np.sum(rho_h) / nc
    #-----------------------------------------------------------------------
    # 0.25-0.75
    x=(width-0.25)/0.5*4-2
    scale =(sp.erf(x)-sp.erf(-2))*0.25
    #-----------------------------------------------------------------------
    return scale
