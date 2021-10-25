import os
import numpy as np
from scipy.interpolate import splrep, splev
from numpy import linalg as LA
from edftpy.utils.common import Field
from edftpy.mpi import sprint
from edftpy.density.density import get_3d_value_recipe
from edftpy.functional.pseudopotential import ReadPseudo

class AtomicDensity(object):
    """
    The densities for atomic atoms.
    """
    def __init__(self, files = None, pseudo = None, **kwargs):
        self._r = {}
        self._arho = {}
        self.direct = False # always use reciprocal method
        self._init_data(files = files, pseudo = pseudo, **kwargs)

    def _init_data(self, files = None, pseudo = None, **kwargs):
        readpp = None
        if files :
            for key, infile in files.items() :
                if not os.path.isfile(infile):
                    raise Exception("Density file " + infile + " for atom type " + str(key) + " not found")
                else:
                    if infile[-4:].lower() == "list" :
                        try :
                            self._r[key], self._arho[key] = self.read_density_list(infile)
                        except Exception :
                            raise Exception("density file '{}' has some problems".format(infile))
                    else :
                        readpp = ReadPseudo(files)
        elif pseudo is not None :
            readpp = pseudo.readpp

        if readpp :
            self._r = readpp._atomic_density_grid
            self._arho = readpp._atomic_density

    def read_density_list(self, infile):
        with open(infile, "r") as fr:
            lines = []
            for i, line in enumerate(fr):
                lines.append(line)

        ibegin = 0
        iend = len(lines)
        data = [line.split()[0:2] for line in lines[ibegin:iend]]
        data = np.asarray(data, dtype = float)

        r = data[:, 0]
        v = data[:, 1]
        return r, v

    def guess_rho(self, ions, grid, ncharge = None, rho = None, dtol=1E-30, **kwargs):
        if len(self._r) == 0 :
            new_rho = self.guess_rho_heg(ions, grid, ncharge, rho, dtol = dtol, **kwargs)
        elif self.direct :
            new_rho = self.guess_rho_atom(ions, grid, ncharge, rho, dtol = dtol, **kwargs)
        else :
            new_rho = self.get_3d_value_recipe(ions, grid.get_reciprocal(), ncharge, rho, dtol = dtol, **kwargs)

        if rho is not None :
            rho[:] = new_rho
        else :
            rho = new_rho
        return rho

    def guess_rho_heg(self, ions, grid, ncharge = None, rho = None, dtol=1E-30, **kwargs):
        """
        """
        if ncharge is None :
            ncharge = 0.0
            for i in range(ions.nat) :
                ncharge += ions.Zval[ions.labels[i]]
        if rho is None :
            rho = Field(grid)
            rho[:] = 1.0
        rho[:] = ncharge / (rho.integral())
        return rho

    def guess_rho_atom(self, ions, grid, ncharge = None, rho = None, dtol=1E-30, **kwargs):
        """
        """
        nr = grid.nr
        dnr = (1.0/nr).reshape((3, 1))
        if rho is None :
            rho = Field(grid) + dtol
        else :
            rho[:] = dtol
        lattice = grid.lattice
        metric = np.dot(lattice.T, lattice)
        latp = np.zeros(3)
        for i in range(3):
            latp[i] = np.sqrt(metric[i, i])
        gaps = latp / nr
        for key in self._r :
            r = self._r[key]
            arho = self._arho[key]
            rcut = np.max(r)
            rcut = min(rcut, 0.5 * np.min(latp))
            rtol = r[0]
            border = (rcut / gaps).astype(np.int32) + 1
            ixyzA = np.mgrid[-border[0]:border[0]+1, -border[1]:border[1]+1, -border[2]:border[2]+1].reshape((3, -1))
            prho = np.zeros((2 * border[0]+1, 2 * border[1]+1, 2 * border[2]+1))
            tck = splrep(r, arho)
            for i in range(ions.nat):
                if ions.labels[i] != key:
                    continue
                prho[:] = 0.0
                posi = ions.pos[i].reshape((1, 3))
                atomp = np.array(posi.to_crys()) * nr
                atomp = atomp.reshape((3, 1))
                ipoint = np.floor(atomp)
                px = atomp - ipoint
                l123A = np.mod(ipoint.astype(np.int32) - ixyzA, nr[:, None])

                positions = (ixyzA + px) * dnr
                positions = np.einsum("j...,kj->k...", positions, grid.lattice)
                dists = LA.norm(positions, axis = 0).reshape(prho.shape)
                index = np.logical_and(dists < rcut, dists > rtol)
                prho[index] = splev(dists[index], tck, der = 0)
                rho[l123A[0], l123A[1], l123A[2]] += prho.ravel()
        if ncharge is None :
            ncharge = 0.0
            for i in range(ions.nat) :
                ncharge += ions.Zval[ions.labels[i]]
        nc = rho.integral()
        sprint('Guess density : ', nc)
        rho[:] *= ncharge / nc
        return rho

    def get_3d_value_recipe(self, ions, grid, ncharge = None, rho = None, dtol=1E-30, direct=True, **kwargs):
        return get_3d_value_recipe(self._r, self._arho, ions, grid, ncharge, rho, dtol, direct, **kwargs)
