import os
import numpy as np
from scipy.interpolate import splrep, splev
from numpy import linalg as LA
# try:
    # from edftpy.io.pp_xml import PPXmlGPAW as PPXml
# except Exception :
    # from edftpy.io.pp_xml import PPXml
# from edftpy.io.pp_xml import PPXml

class AtomicDensity(object):
    """
    LocalPseudo class handles local pseudo potentials.
    This is a template class and should never be touched.
    """

    def __init__(self, files =None, grid=None, ions=None, ftypes = None, **kwargs):

        self.files = files
        if self.files is None:
            return
        # if files None:
            # raise AttributeError("Must specify files for atomic density")

        self._r = {}
        self._arho = {}
        self._info= {}
        for key, infile in files.items() :
            print("setting key: " + key)
            if not os.path.isfile(infile):
                raise Exception("Density file " + infile + " for atom type " + str(key) + " not found")
            else:
                if ftypes is not None :
                    ftype = ftypes[key]
                else :
                    if infile[-3:].lower() == "xml" :
                        ftype = "xml"
                    if infile[-6:].lower() == "recpot":
                        ftype = "recpot"
                    elif infile[-3:].lower() == "upf":
                        ftype = "upf"
                    elif infile[-3:].lower() == "psp" or infile[-4:].lower() == "psp8":
                        ftype = "psp"
                        raise Exception("density file not supported")
                    else :
                        ftype = "list"

            if ftype == 'psp' :
                self._r[key], self._arho[key], self._info[key] = self.read_density_psp(infile)
            elif ftype == 'xml' :
                self._r[key], self._arho[key], self._info[key] = self.read_density_xml(infile)
            elif ftype == 'list' :
                try :
                    self._r[key], self._arho[key], self._info[key] = self.read_density_list(infile)
                except Exception :
                    raise Exception("density file have some problems")
            else :
                    raise Exception("density file not supported")

    def read_density_psp(self, infile):
        with open(infile, "r") as fr:
            lines = []
            for i, line in enumerate(fr):
                if i > 5 :
                    line = line.replace('D', 'E')
                lines.append(line)
        info = {}

        # line 2 :atomic number, pseudoion charge, date
        values = lines[1].split()
        atomicnum = int(float(values[0]))
        Zval = float(values[1])
        # line 3 :pspcod,pspxc,lmax,lloc,mmax,r2well
        values = lines[2].split()
        if int(values[0]) != 8 :
            raise AttributeError("Only support psp8 format pseudopotential with psp")
        info['atomicnum'] = atomicnum
        info['Zval'] = Zval
        info['pspcod'] = 8
        info['pspxc'] = int(values[1])
        info['lmax'] = int(values[2])
        info['lloc'] = int(values[3])
        info['r2well'] = int(values[5])
        info['mmax'] = int(values[4])
        mmax = info['mmax']
        lloc = info['lloc']

        ibegin = 6+ (mmax + 1) * lloc + mmax
        iend = ibegin + mmax
        data = [line.split()[1:3] for line in lines[ibegin:iend]]
        data = np.asarray(data, dtype = float)

        r = data[:, 0] 
        v = data[:, 1]/(4.0 * np.pi)
        # v = data[:, 1]
        # print('r, v', r, v)
        return r, v, info

    def read_density_xml(self, infile):
        # try:
            # from edftpy.io.pp_xml import PPXmlGPAW as PPXml
        # except Exception :
            # from edftpy.io.pp_xml import PPXml
        from edftpy.io.pp_xml import PPXml
        pp = PPXml(infile)
        r, v, info = pp.get_pseudo_valence_density()
        return r, v/np.sqrt(4*np.pi), info

    def read_density_list(self, infile):
        with open(infile, "r") as fr:
            lines = []
            for i, line in enumerate(fr):
                lines.append(line)
        info = {}

        ibegin = 0
        iend = len(lines)
        data = [line.split()[0:2] for line in lines[ibegin:iend]]
        data = np.asarray(data, dtype = float)

        r = data[:, 0] 
        v = data[:, 1]
        return r, v, info

    def guess_rho(self, ions, grid, ncharge = None, rho = None, ndens = 2, **kwargs):
        if self.files is None :
            rho = self.guess_rho_heg(ions, grid, ncharge, rho, **kwargs)
        else :
            rho = self.guess_rho_atom(ions, grid, ncharge, rho, ndens, **kwargs)
        return rho

    def guess_rho_heg(self, ions, grid, ncharge = None, rho = None, ndens = 2, **kwargs):
        """
        """
        nr = grid.nr
        if rho is None :
            rho = np.zeros(nr)

        if ncharge is None :
            ncharge = 0.0
            for i in range(ions.nat) :
                ncharge += ions.Zval[ions.labels[i]]
        rho[:] = ncharge / (np.size(rho) * grid.dV)
        return rho

    def guess_rho_atom(self, ions, grid, ncharge = None, rho = None, ndens = 2, **kwargs):
        """
        """
        nr = grid.nr
        dnr = (1.0/nr).reshape((3, 1))
        dtol = 1E-10 # for safe, replace the zero density as a value
        if rho is None :
            rho = np.ones(nr) * dtol
        else :
            rho[:] = np.ones(nr) * dtol
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
        print('Guess density : ', np.sum(rho) * grid.dV)
        rho[:] *= ncharge / (np.sum(rho) * grid.dV)
        return rho

    def guess_rho_all(self, ions, grid, pbc = [1, 1, 1], **kwargs):
        """
        haven't finished
        """
        nr = grid.nr
        dnr = (1.0/nr).reshape((3, 1))
        dtol = 1E-8 # for safe, replace the zero density as a value
        rho = np.ones(nr) * dtol
        lattice = grid.lattice
        metric = np.dot(lattice.T, lattice)
        latp = np.zeros(3)
        for i in range(3):
            latp[i] = np.sqrt(metric[i, i])
        gaps = latp / nr
        #-----------------------------------------------------------------------
        cellbound = np.empty((2, 3))
        cellbound[0, :] = 0.0
        cellbound[1, :] = latp.copy()
        #-----------------------------------------------------------------------
        for key in self._r :
            r = self._r[key]
            arho = self._arho[key]
            rcut = np.max(r)
            rtol = r[0]
            border = (rcut / gaps).astype(np.int32) + 1
            ixyzA = np.mgrid[-border[0]:border[0]+1, -border[1]:border[1]+1, -border[2]:border[2]+1].reshape((3, -1))
            prho = np.zeros((2 * border[0]+1, 2 * border[1]+1, 2 * border[2]+1))
            tck = splrep(r, arho)
            #-----------------------------------------------------------------------
            n = np.ceil(rcut / latp).astype(np.int)
            pbcmap = np.zeros((2, 3), dtype=np.int32)
            pbcmap[0, :] = 0
            pbcmap[1, :] = 2 * n[:] + 1
            rpbc = np.empty((2 * n[0] + 1, 2 * n[1] + 1, 2 * n[2] + 1, 3))
            for ix in np.arange(-n[0], n[0] + 1):
                for iy in np.arange(-n[1], n[1] + 1):
                    for iz in np.arange(-n[2], n[2] + 1):
                        r = np.einsum("j,ij->i", np.array([ix, iy, iz], dtype=np.float), lattice)
                        rpbc[ix + n[0], iy + n[1], iz + n[2], :] = r
            #-----------------------------------------------------------------------
            for i in range(ions.nat):
                if ions.labels[i] != key:
                    continue
                #-----------------------------------------------------------------------
                posi = ions.pos[i].reshape((1, 3))
                lbound = posi - rcut
                ubound = posi + rcut
                for j in range(3):
                    if lbound[0, j] < cellbound[0, j]:
                        pbcmap[1, j] = 2 * n[j] + 1
                    else:
                        pbcmap[1, j] = n[j] + 1

                    if ubound[0, j] > cellbound[1, j]:
                        pbcmap[0, j] = 0
                    else:
                        pbcmap[0, j] = n[j]
                for i0 in range(pbcmap[0, 0], pbcmap[1, 0]):
                    for i1 in range(pbcmap[0, 1], pbcmap[1, 1]):
                        for i2 in range(pbcmap[0, 2], pbcmap[1, 2]):
                            pbcpos = posi + rpbc[i0, i1, i2, :]
                            #-----------------------------------------------------------------------
                            atomp = np.array(pbcpos.to_crys()) * nr
                            atomp = atomp.reshape((3, 1))
                            ipoint = np.floor(atomp) 
                            px = atomp - ipoint
                            l123A = np.mod(ipoint.astype(np.int32) - ixyzA, nr[:, None])

                            positions = (ixyzA - px) * dnr
                            positions = np.einsum("j...,kj->k...", positions, grid.lattice)
                            dists = LA.norm(positions, axis = 0).reshape(prho.shape)
                            index = np.logical_and(dists < rcut, dists > rtol)
                            prho[index] = splev(dists[index], tck, der = 0)
                            rho[l123A[0], l123A[1], l123A[2]] += prho.ravel()
        return rho
