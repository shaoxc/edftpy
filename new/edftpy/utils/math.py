import numpy as np
import scipy.special as sp
from scipy import ndimage, signal
from functools import reduce
import itertools
from dftpy.utils import grid_map_index, grid_map_data

from .common import Field, Grid


def gaussian(x, sigma = 0.4, mu = 0.0, dim = 3):
    if dim > 1 :
        y = 1.0/(np.sqrt(2.0 * np.pi) * sigma) ** dim * np.exp((x - mu) ** 2 /(-2.0 * sigma * sigma))
    elif dim == 1 :
        y = 1.0/(np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu)/sigma) ** 2.0)
    else : # Just for debug
        y = 1.0/(np.sqrt(2.0 * np.pi * sigma)) ** 3 * np.exp((x - mu) ** 2 /(-2.0 * sigma))
    return y

def gaussian_g(x, sigma = 0.1, dim = 3, kind = 0):
    if kind > 0 :
        y = (np.sqrt(2.0 * np.pi) * sigma) ** (dim - 1) * np.exp(-2 * (x * sigma * np.pi) ** 2)
    else :
        y = (np.sqrt(2.0 * np.pi) * sigma) ** (dim - 1) * np.exp(-0.5 * (x * sigma) ** 2)
    return y

def fermi_dirac(x, mu = None, kt = None):
    if mu is None :
        mu = 2.0/3.0 * np.max(x)
    if kt is None :
        kt = mu * 0.1
    f = np.exp((x - mu)/kt) + 1.0
    f = 1.0/f
    return f

def smooth_interpolating_potential(density, potential, a = 5E-2, b = 3):
    mask1 = np.logical_and(density > 0, density < a)
    mask2 = density >= a
    fab = np.zeros_like(potential)
    fab[mask1] = (1.0+np.exp(a/(a - density[mask1])))/(np.exp(a/density[mask1])+np.exp(a/(a - density[mask1])))
    fab[mask2] = 1.0
    # fab = sp.erfc(density/a)
    potential *= fab
    return potential

def interpolation_3d(data, nr = None, direct = True, index = None, grid = None):
    if nr is None :
        nr = grid.nr
    nr0 = np.array(data.shape)
    h = 1.0/nr
    x, y, z = np.mgrid[0:1:h[0], 0:1:h[1], 0:1:h[2]]
    x *= nr0[0]
    y *= nr0[1]
    z *= nr0[2]
    results = ndimage.map_coordinates(data, (x, y, z), order=3, mode="nearest")
    if grid is None :
        grid = Grid(data.grid.lattice, nr, direct = True)
    results = Field(grid, data=results)
    return results

def union_mlist(arrs, keys = None, array = False):
    if keys is None :
        keys = set(itertools.chain.from_iterable(arrs))
    for key in keys:
        comp = [(i, item) for i, item in enumerate(arrs) if key in item]
        ind, comp = zip(*comp)
        ind = set(ind)
        if len(comp) > 0 :
            arrs = [arrs[i] for i, item in enumerate(arrs) if i not in ind]
            if len(comp) > 1 :
                if array :
                    arrs.append(reduce(np.union1d, comp))
                else :
                    arrs.append(set(itertools.chain.from_iterable(comp)))
            else :
                arrs.append(comp[0])
    return arrs

def dict_update(d, u):
    """
    ref :
        https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    import collections.abc
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d