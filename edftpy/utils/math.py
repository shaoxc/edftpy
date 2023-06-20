import numpy as np
from numpy import linalg as LA
import scipy.special as sp
from scipy import ndimage, signal
from functools import reduce
import itertools
import hashlib
from ase.symbols import Symbols, symbols2numbers

from dftpy.utils import grid_map_index, grid_map_data

from .common import Field, Grid


def gaussian(x, sigma = 0.4, mu = 0.0, dim = 3, deriv = 0):
    if dim > 1 :
        y = 1.0/(np.sqrt(2.0 * np.pi) * sigma) ** dim * np.exp((x - mu) ** 2 /(-2.0 * sigma * sigma))
    elif dim == 1 :
        y = 1.0/(np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu)/sigma) ** 2.0)
    else : # Just for debug
        y = 1.0/(np.sqrt(2.0 * np.pi * sigma)) ** 3 * np.exp((x - mu) ** 2 /(-2.0 * sigma))
    if deriv == 0 :
        pass
    elif deriv == 1 :
        y *= (mu-x)/sigma**2
    else :
        raise AttributeError(f"Sorry, the gaussian not support 'deriv' with {deriv}")
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

def union_mlist_u(arrs, keys = None, array = False):
    if keys is None :
        keys = set(itertools.chain.from_iterable(arrs))
    for key in keys:
        comp = []
        for i, item in enumerate(arrs):
            if key in item :
                arrs.pop(i)
                comp.append(item)
        nl = len(comp)
        if nl > 1 :
            if array :
                arrs.append(reduce(np.union1d, comp))
            else :
                arrs.append(set(itertools.chain.from_iterable(comp)))
        elif nl == 1 :
            arrs.append(comp[0])
    return arrs

def union_mlist(arrs, keys = None, array = False, number = True):
    if number :
        return union_mlist_number(arrs, keys=keys, array=array)
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
    for i, item in enumerate(arrs): arrs[i] = list(item)
    return arrs

def union_mlist_number(arrs, keys = None, array = False):
    if keys is None :
        keys = set(itertools.chain.from_iterable(arrs))
    sub_inds = arrs.copy()
    for ik, key in enumerate(keys):
        comp = []
        for i in sub_inds[key] :
            comp.append(arrs[i])
        nl = len(comp)
        if nl > 1 :
            if array :
                v = reduce(np.union1d, comp)
            else :
                v = set(itertools.chain.from_iterable(comp))
        else :
            v = comp[0]
        for i in sub_inds[key] :
            arrs[i] = v

        if ik + 1 < len(keys) :
            key = keys[ik + 1]
            if array :
                v_new = reduce(np.union1d, [v, sub_inds[key]])
            else :
                v_new = set(itertools.chain.from_iterable([v, sub_inds[key]]))
            if len(v_new) < len(sub_inds[key]) + len(v):
                sub_inds[key] = v_new

    used = []
    values = []
    for item in arrs:
        for x in item : break
        if x in used : continue
        if array :
            values.append(item)
        else :
            values.append(np.asarray(sorted(item)))
        used.extend(item)
    return values

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

def get_core_points(pos, grid, rcut):
    """get_core_points.

    Parameters
    ----------
    pos : np.ndarray (3,)
        scaled position
    grid : Grid
        grid
    rcut : float
        distance cutoff
    """
    nr = grid.nrR
    dnr = (1.0/nr).reshape((3, 1))
    gaps = grid.spacings
    border = (rcut / gaps).astype(np.int32) + 1
    border = np.minimum(border, nr//2)
    ixyzA = np.mgrid[-border[0]:border[0]+1, -border[1]:border[1]+1, -border[2]:border[2]+1].reshape((3, -1))
    atomp = nr * pos
    atomp = atomp.reshape((3, 1))
    ipoint = np.floor(atomp + 1E-8)
    px = atomp - ipoint
    l123A = np.mod(ipoint.astype(np.int32) - ixyzA, nr[:, None])
    positions = (ixyzA + px) * dnr
    positions = np.einsum("j...,kj->k...", positions, grid.lattice)
    nr_orth = (2 * border[0]+1, 2 * border[1]+1, 2 * border[2]+1)
    dists = LA.norm(positions, axis = 0).reshape(nr_orth)
    mask = get_grid_array_mask(grid, l123A)
    index =(dists[mask] < rcut).ravel()
    inds = (l123A[0][index], l123A[1][index], l123A[2][index])
    return inds

def get_grid_array_mask(grid, l123A):
    if grid.mp.comm.size == 1 :
        return slice(None)
    offsets = grid.offsets.reshape((3, 1))
    nr = grid.nr
    #-----------------------------------------------------------------------
    l123A -= offsets
    mask = np.logical_and(l123A[0] > -1, l123A[0] < nr[0])
    mask1 = np.logical_and(l123A[1] > -1, l123A[1] < nr[1])
    np.logical_and(mask, mask1, out = mask)
    np.logical_and(l123A[2] > -1, l123A[2] < nr[2], out = mask1)
    np.logical_and(mask, mask1, out = mask)
    #-----------------------------------------------------------------------
    return mask

def get_hash(x):
    value = hashlib.md5(np.array(sorted(x))).hexdigest()
    return value

def get_formal_charge(numbers = None, symbols = None, data = {}, method = 'simple'):
    if len(data) == 0 :
        return 0
    if method == 'simple' :
        if numbers is None :
            numbers = Symbols.fromsymbols(symbols).numbers
        else :
            numbers = list(numbers)
        charge = get_formal_charge_simple(numbers, data)
    else :
        raise AttributeError("Sorry, not support '{}' method yet.".format(method))
    return charge

def get_formal_charge_simple(numbers, data):
    charge = 0
    ref = sorted(list(data.items()),key=lambda x: len(x[0]), reverse=True)
    for i, item in enumerate(ref):
        numb = symbols2numbers(item[0])
        ref[i] = [numb, item[1]]
    rest = len(numbers)
    rest_prev = rest
    for item in ref :
        while rest > 0 :
            flag = True
            for i in item[0] :
                if i not in numbers :
                    flag = False
                    break
            if flag :
                charge += item[1]
                for i in item[0] :
                    numbers.remove(i)
                rest = len(numbers)
            if rest < rest_prev :
                rest_prev = rest
            else :
                break
        if rest == 0 :
            break
    else :
        raise AttributeError("Can not find all the molecules : {}".format(numbers))
    return charge

def copy_array(arr, out = None, add = False, index = None):
    if index is None : index = slice(None)
    if out is None :
        out = arr.copy()
    elif add :
        out[index] += arr
    else :
        out[index] = arr
    return out
