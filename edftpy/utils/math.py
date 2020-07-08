import numpy as np
import scipy.special as sp
from scipy import ndimage
from scipy import signal
from dftpy.utils import grid_map_index, grid_map_data

from .common import Field, Grid


def gaussian(x, sigma = 0.4, mu = 0.0, dim = 3):
    if dim > 1 :
        y = 1.0/(np.sqrt(2.0 * np.pi) * sigma) ** dim * np.exp((x - mu) ** 2 /(-2.0 * sigma * sigma))
    elif dim == 1 :
        y = 1.0/(np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu)/sigma) ** 2.0)
    else : # Just for debug
        y = 1.0/(np.sqrt(2.0 * np.pi * sigma)) ** 3 * np.exp((x - mu) ** 2 /(-2.0 * sigma))
        # y = np.zeros_like(x)
    return y

def fermi_dirac(x, mu = None, kt = None):
    if mu is None :
        mu = 2.0/3.0 * np.max(x)
    if kt is None :
        kt = mu * 0.1
    f = np.exp((x - mu)/kt) + 1.0
    f = 1.0/f
    return f

def coarse_to_fine_bak(data, nr_fine, direct = True):
    if hasattr(data, 'fft'):
        value = data.fft()
    else :
        value = data
    nr = value.shape
    nr2 = nr_fine.copy()
    nr2[2] = nr2[2]//2+1
    index = grid_map_index(nr, nr2)
    grid = Grid(value.grid.lattice, nr_fine, direct = False)
    value_fine = Field(grid, direct = False)
    value_fine[index[0], index[1], index[2]] = value.ravel()
    if direct :
        results = value_fine.ifft(force_real=True)
    else :
        results = value_fine
    return results

def fine_to_coarse_bak(data, nr_coarse, direct = True):
    if hasattr(data, 'fft'):
        value = data.fft()
    else :
        value = data
    nr = value.shape
    nr2 = nr_coarse.copy()
    nr2[2] = nr2[2]//2+1
    index = grid_map_index(nr2, nr)
    value_coarse = value[index[0], index[1], index[2]]
    grid = Grid(value.grid.lattice, nr_coarse, direct = False)
    value_g = Field(grid, data=value_coarse, direct=False)
    if direct :
        results = value_g.ifft(force_real=True)
    else :
        results = value_g
    return results

def smooth_interpolating_potential(density, potential, a = 5E-2, b = 3):
    mask1 = np.logical_and(density > 0, density < a)
    mask2 = density >= a
    fab = np.zeros_like(potential)
    fab[mask1] = (1.0+np.exp(a/(a - density[mask1])))/(np.exp(a/density[mask1])+np.exp(a/(a - density[mask1])))
    fab[mask2] = 1.0
    # fab = sp.erfc(density/a)
    potential *= fab
    return potential
