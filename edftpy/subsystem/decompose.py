import numpy as np
from ase.neighborlist import neighbor_list

from dftpy.formats.ase_io import ions2ase

from edftpy.utils.math import union_mlist

def from_distance_to_sub(ions, cutoff = 3, max_nbins=1e6, **kwargs):
    atoms = ions2ase(ions)
    nat = atoms.get_global_number_of_atoms()
    inda, indb, dists = neighbor_list('ijd', atoms, cutoff, self_interaction=True, max_nbins=max_nbins)
    # maskr = np.zeros(nat, dtype = 'bool')
    subcells = []
    index = []
    for i in range(nat):
        index.append(i)
        firsts = np.where(inda == i)[0]
        neibors = indb[firsts]
        subcells.append(neibors)
    keys = np.arange(nat)
    subcells = union_mlist(subcells, keys = keys, array = True)
    return subcells
