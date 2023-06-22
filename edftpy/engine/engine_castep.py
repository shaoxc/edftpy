import caspytep
import numpy as np
import ase.io.castep as ase_io_driver
from ase.calculators.castep import Castep as ase_calc_driver

from dftpy.constants import LEN_CONV

from edftpy.io import ions2ase
from edftpy.engine.engine import Engine
from edftpy.utils.common import Grid, Field, Atoms
from edftpy.mpi import SerialComm

try:
    __version__ = caspytep.__version__
except Exception :
    __version__ = '0.0.1'

class EngineCastep(Engine):
    def __init__(self, **kwargs):
        unit_len = 1.0
        unit_vol = unit_len ** 3
        units = kwargs.get('units', {})
        kwargs['units'] = units
        kwargs['units']['length'] = unit_len
        kwargs['units']['volume'] = unit_vol
        kwargs['units']['energy'] = 1.0
        kwargs['units']['order'] = 'F'
        super().__init__(**kwargs)
        self.embed = None
        self.comm = SerialComm()

    def get_force(self, icalc = 3, **kwargs):
        extpot, extene = self._get_extpot(self.mdl.den, self.grid)
        #-----------------------------------------------------------------------
        # labels = self.subcell.ions.labels
        labels = self.subcell.ions.Z
        u, index, counts = np.unique(labels, return_index=True, return_counts=True)
        sidx = np.argsort(index)
        nats = counts[sidx]
        #-----------------------------------------------------------------------
        # nats = []
        # keys = list(set(labels))
        # keys.sort(key = labels.index)
        # for key in keys :
            # nats.append(labels.count(key))
        #-----------------------------------------------------------------------
        ntyp = len(nats)
        nmax = max(nats)
        fs = np.zeros((3, nmax, ntyp), order = 'F')
        caspytep.firstd.firstd_calculate_forces_edft_ext(self.mdl, fs, extpot, extene, self.exttype, icalc)
        forces = np.empty((self.subcell.ions.nat, 3))
        n = 0
        for i, item in enumerate(nats):
            forces[n:n+item] = fs[:, :item, sidx[i]].T
            n += item
        return forces

    def embed_base(self, exttype = 0, diag_conv = 1E-1, lewald = False, iterative = True, embed = None, **kwargs):
        embed.exttype = exttype
        embed.diag_conv = diag_conv
        # Include ewald or not
        embed.lewald = lewald
        embed.iterative = iterative
        embed.tddft.iterative = iterative
        self.embed = embed
        return embed

    def calc_energy(self, etype = 1, **kwargs):
        energy = 0.0
        if etype == 1 :
            kinetic_energy = caspytep.electronic.electronic_get_energy('kinetic_energy')
            nonlocal_energy = caspytep.electronic.electronic_get_energy('nonlocal_energy')
            ts = caspytep.electronic.electronic_get_energy('-TS')

            locps_energy = caspytep.electronic.electronic_get_energy('locps_energy')
            ion_noncoulomb_energy = caspytep.electronic.electronic_get_energy('ion_noncoulomb_energy')

            hartree_energy = caspytep.electronic.electronic_get_energy('hartree_energy')

            xc_energy = caspytep.electronic.electronic_get_energy('xc_energy')
            energy += kinetic_energy + nonlocal_energy + ts
            if self.embed.exttype & 1 == 0 :
                energy += locps_energy + ion_noncoulomb_energy
            if self.embed.exttype & 2 == 0 :
                energy += hartree_energy
            if self.embed.exttype & 4 == 0 :
                energy += xc_energy
        else :
            # This way need give exact energy of external potential (extene)
            #-----------------------------------------------------------------------
            total_energy = caspytep.electronic.electronic_get_energy('total_energy')
            ion_ion_energy0 = caspytep.electronic.electronic_get_energy('ion_ion_energy0')
            locps_energy = caspytep.electronic.electronic_get_energy('locps_energy')
            hartree_energy = caspytep.electronic.electronic_get_energy('hartree_energy')
            xc_energy = caspytep.electronic.electronic_get_energy('xc_energy')
            ion_noncoulomb_energy = caspytep.electronic.electronic_get_energy('ion_noncoulomb_energy')
            # energy += total_energy - ion_ion_energy0 - locps_energy - ion_noncoulomb_energy - hartree_energy
            energy += total_energy - ion_ion_energy0
            if self.embed.exttype & 1 == 1 :
                # print('ks 1', locps_energy + ion_noncoulomb_energy)
                energy -= (locps_energy + ion_noncoulomb_energy)
            if self.embed.exttype & 2 == 2 :
                energy -= hartree_energy
                # print('ks 2', hartree_energy)
            if self.embed.exttype & 4 == 4 :
                energy -= xc_energy
                # print('ks 4', xc_energy)
        energy *= self.units['energy']
        return energy

    def get_grid(self, nr = None, **kwargs):
        current_basis = caspytep.basis.get_current_basis()
        nx = current_basis.ngx
        ny = current_basis.ngy
        nz = current_basis.ngz
        if nr is None :
            nr = np.array([nx, ny, nz])
        else :
            nr[:] = np.array([nx, ny, nz])
        return nr

    def get_rho(self, rho, **kwargs):
        rho[:] = self.mdl.den.real_charge

    def get_rho_core(self, rho, **kwargs):
        pass

    def get_ef(self, **kwargs):
        return self.mdl.fermi_energy

    def _initial_files(self, inputfile = None, comm = None, **kwargs):
        if hasattr(comm, 'py2f') :
            commf = comm.py2f()
        else :
            commf = None

        self.comm = comm or self.comm
        self.commf = commf

        if inputfile is None :
            prefix = kwargs.get('prefix', 'sub_qe')
            inputfile = prefix + '.in'
            if self.comm.rank == 0 :
                self.write_input(inputfile, **kwargs)
            if self.comm.size > 1 : self.comm.Barrier()
        self.inputfile = inputfile

    def initial(self, inputfile = None, comm = None, iterative = True, **kwargs):
        pass

    def tddft_initial(self, inputfile = None, comm = None, **kwargs):
        pass

    def save(self, save = ['D'], **kwargs):
        pass

    def scf(self, **kwargs):
        self.embed.mix_coef = -1.0
        qepy.qepy_electrons_scf(0, 0, self.embed)

    def scf_mix(self, coef = 0.7, **kwargs):
        self.embed.mix_coef = coef
        qepy.qepy_electrons_scf(2, 0, self.embed)

    def set_extpot(self, extpot, **kwargs):
        extpot = extpot / self.units['energy']
        qepy.qepy_mod.qepy_set_extpot(self.embed, extpot)

    def set_rho(self, rho, **kwargs):
        pass

    def stop_scf(self, status = 0, save = ['D'], **kwargs):
        pass

    def stop_tddft(self, status = 0, save = ['D'], **kwargs):
        pass

    def end_scf(self, **kwargs):
        pass

    def end_tddft(self, **kwargs):
        pass

    def tddft(self, **kwargs):
        pass

    def tddft_after_scf(self, inputfile = None, **kwargs):
        pass

    def update_ions(self, subcell, update = 0, **kwargs):
        pass

    def get_dnorm(self, **kwargs):
        return self.embed.dnorm

    def write_input(self, prefix = 'sub_driver', subcell = None, params = {}, cell_params = {}, base_in_file = None, **kwargs):
        self.subcell = subcell
        ase_atoms = self._build_ase_atoms(params, cell_params, base_in_file)
        self._write_cell(self.prefix + '.cell', ase_atoms, params = cell_params)
        self._write_params(self.prefix + '.param', ase_atoms, params = params)

    def _build_ase_atoms(self, params = None, cell_params = None, base_in_file = None):
        ase_atoms = ions2ase(self.subcell.ions)
        ase_atoms.set_calculator(ase_calc_driver())
        ase_cell = ase_atoms.calc.cell
        if cell_params is not None :
            for k1, v1 in cell_params.items() :
                if isinstance(v1, dict):
                    value = []
                    for k2, v2 in v1.items() :
                        value.append((k2, v2))
                    ase_cell.__setattr__(k1, value)
                else :
                    ase_cell.__setattr__(k1, v1)

        if base_in_file is not None :
            fileobj = open(base_in_file, 'r')
            calc = ase_io_driver.read_param(fd=fileobj)
            fileobj.close()
            ase_atoms.calc.param = calc.param

        driver_params = ase_atoms.calc.param

        if params is not None :
            for k1, v1 in params.items() :
                if isinstance(v1, dict):
                    value = []
                    for k2, v2 in v1.items() :
                        value.append((k2, v2))
                    driver_params.__setattr__(k1, value)
                else :
                    driver_params.__setattr__(k1, v1)

    def _write_cell(self, outfile, ase_atoms, pbc = None, params = None, **kwargs):
        ase_io_driver.write_cell(outfile, ase_atoms, force_write = True)
        return

    def _write_params(self, outfile, ase_atoms, params = None, **kwargs):
        driver_params = ase_atoms.calc.param
        ase_io_driver.write_param(outfile, driver_params, force_write = True)
        return
