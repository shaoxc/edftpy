import numpy as np
from ..mixer import LinearMixer
from .driver import Driver
from .hamiltonian import Hamiltonian


class DFTpyKS(Driver):
    """
    Not finish !!!
    """
    def __init__(self, evaluator = None, nbnd = None, options = None, **kwargs):
        Driver.__init__(self, options = options)
        self.evaluator = evaluator
        self.hamiltonian = None
        self.nbnd = nbnd
        self.rho = None
        self.nocc = 2 # !!! need modified later
        self.wfs = None
        self.occupations = None
        self.eigs = None
        self.fermi = None
        self.mixer = LinearMixer()

    def _build_hamiltonian(self, density, vext = None, **kwargs):
        func = self.evaluator(density, calcType = ['V'])
        vpot = func.potential
        if vext is not None :
            vpot += vext
        # print('vpot', vpot)
        self.hamiltonian = Hamiltonian(vpot)
        self.N = density.integral()
        self.rho = density
        if self.nbnd is None :
            self.nbnd = int(self.N/self.nocc) + 2

    def _get_wfs(self):
        # print(self.nbnd, 'self.nbnd')
        eigs, psi_list = self.hamiltonian.diagonize(self.nbnd)
        self.eigs = eigs
        self.wfs = psi_list

    def _get_occupations(self):
        self.occupations = np.zeros(self.nbnd)
        ik = int(self.N/self.nocc)
        self.occupations[:ik] = self.nocc
        self.occupations[ik] = self.N - np.sum(self.occupations[:ik])

    def _get_pseudo_density(self):
        rho = None
        for wf, occ in zip(self.wfs, self.occupations) :
            if rho is None :
                rho = np.real(np.conj(wf)*wf)*occ
            else :
                rho += np.real(np.conj(wf)*wf)*occ
        return rho

    def _get_fermi(self):
        a = np.where(self.occupations < self.nocc)
        if len(a[0]) < 1 :
            ik = len(self.nbnd)
        else :
            ik = int(a[0][0])
        homo = self.eigs[ik - 1]
        lomo = self.eigs[ik]
        self.fermi = 0.5 * (homo + lomo)

    def get_density(self, density, vext = None, **kwargs):
        self._build_hamiltonian(density, vext = vext, **kwargs)
        self._get_wfs()
        self._get_occupations()
        self._get_fermi()
        rho = self._get_pseudo_density()
        return rho

    def get_kinetic_energy(self, **kwargs):
        energy = 0.0
        for wf, occ in zip(self.wfs, self.occupations) :
            energy += -0.5 * (wf.laplacian(force_real = True) * wf).integral()*occ
        return energy

    def get_energy(self, density, **kwargs):
        func = self.evaluator(density, calcType = ['E'])
        # print('pot',func.energy * 27.2113834279111)
        # print('ke',self.get_kinetic_energy() * 27.2113834279111)
        energy = func.energy + self.get_kinetic_energy()
        return energy

    def get_energy_potential(self, density, calcType = ['E', 'V'], **kwargs):
        func = self.evaluator(density, calcType = calcType)
        if 'E' in calcType :
            func.energy += self.get_kinetic_energy()
        return func

    def update_density(self, prev, new, **kwargs):
        results = self.mixer(prev, new, coef=0.1)
        return results

    def get_fermi_level(self, **kwargs):
        results = self.fermi
        return results
