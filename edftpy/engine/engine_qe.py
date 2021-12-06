import qepy
import numpy as np
import os
import ase.io.espresso as ase_io_driver
from ase.calculators.espresso import Espresso as ase_calc_driver
from collections import OrderedDict

from dftpy.constants import LEN_CONV

from edftpy.io import ions2ase
from edftpy.engine.engine import Engine
from edftpy.utils.common import Grid, Field, Atoms
from edftpy.mpi import SerialComm

try:
    __version__ = qepy.__version__
except Exception :
    __version__ = '0.0.1'

class EngineQE(Engine):
    def __init__(self, **kwargs):
        unit_len = LEN_CONV["Angstrom"]["Bohr"] / qepy.constants.ANGSTROM_AU
        units = kwargs.get('units', {})
        units['length'] = kwargs.get('length', unit_len)
        units['energy'] = kwargs.get('energy', 0.5)
        units['order'] = 'F'
        kwargs['units'] = units
        super().__init__(**kwargs)
        self.embed = None
        self.comm = SerialComm()

    def get_force(self, icalc = 3, **kwargs):
        qepy.qepy_forces(icalc)
        forces = qepy.force_mod.get_array_force().T * self.units['energy']/self.units['length']
        return forces

    def embed_base(self, exttype = 0, diag_conv = 1E-1, lewald = False, iterative = True, **kwargs):
        embed = qepy.qepy_common.embed_base()
        embed.exttype = exttype
        embed.diag_conv = diag_conv
        # Include ewald or not
        embed.lewald = lewald
        embed.iterative = iterative
        embed.tddft.iterative = iterative
        self.embed = embed
        return embed

    def calc_energy(self, **kwargs):
        qepy.qepy_calc_energies(self.embed)
        energy = self.embed.etotal * self.units['energy']
        return energy

    def get_energy(self, olevel = 0, **kwargs):
        if olevel == 0 :
            qepy.qepy_calc_energies(self.embed)
        energy = self.embed.etotal * self.units['energy']
        return energy

    def clean_saved(self, *args, **kwargs):
        qepy.qepy_clean_saved()

    def get_grid(self, **kwargs):
        nr = qepy.qepy_mod.qepy_get_grid()
        return nr

    def get_rho(self, rho, **kwargs):
        qepy.qepy_mod.qepy_get_rho(rho)

    def get_rho_core(self, rho, **kwargs):
        qepy.qepy_mod.qepy_get_rho_core(rho)

    def get_ef(self, **kwargs):
        return qepy.ener.get_ef()

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

    def initial(self, inputfile = None, comm = None, **kwargs):
        self._initial_files(inputfile = inputfile, comm = comm, **kwargs)
        qepy.qepy_pwscf(self.inputfile, self.commf)
        self.embed = self.embed_base(**kwargs)

    def tddft_initial(self, inputfile = None, comm = None, **kwargs):
        self._initial_files(inputfile = inputfile, comm = comm, **kwargs)
        qepy.qepy_tddft_main_initial(self.inputfile, self.commf)
        qepy.read_file()
        self.embed = self.embed_base(**kwargs)

    def save(self, save = ['D'], **kwargs):
        if 'W' in save :
            what = 'all'
        else :
            what = 'config-nowf'
        qepy.punch(what)
        qepy.close_files(False)

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
        qepy.qepy_mod.qepy_set_rho(rho)

    def stop_scf(self, status = 0, save = ['D'], **kwargs):
        if 'W' in save :
            what = 'all'
        else :
            what = 'config-nowf'
        qepy.qepy_stop_run(status, what = what)
        qepy.qepy_mod.qepy_close_stdout('')
        qepy.qepy_clean_saved()

    def stop_tddft(self, status = 0, save = ['D'], **kwargs):
        qepy.qepy_stop_tddft(status)
        qepy.qepy_mod.qepy_close_stdout('')
        qepy.qepy_clean_saved()

    def end_scf(self, **kwargs):
        if self.embed.iterative :
            self.embed.finish = True
            qepy.qepy_electrons_scf(0, 0, self.embed)

    def end_tddft(self, **kwargs):
        if self.embed.tddft.iterative :
            self.embed.tddft.finish = True
            qepy.qepy_molecule_optical_absorption(self.embed)

    def tddft(self, **kwargs):
        qepy.qepy_molecule_optical_absorption(self.embed)

    def tddft_after_scf(self, inputfile = None, **kwargs):
        inputfile = inputfile or self.inputfile
        qepy.qepy_tddft_readin(inputfile)
        qepy.qepy_tddft_main_setup(self.embed)

    def update_ions(self, subcell, update = 0, **kwargs):
        pos = subcell.ions.pos.to_cart().T
        if hasattr(qepy, 'qepy_api'):
            qepy.qepy_api.qepy_update_ions(self.embed, pos, update)
        else : # old version
            qepy.qepy_mod.qepy_update_ions(self.embed, pos, update)

    def wfc2rho(self, *args, **kwargs):
        qepy.qepy_tddft_mod.qepy_cetddft_wfc2rho()

    def get_dnorm(self, **kwargs):
        return self.embed.dnorm

    def write_input(self, filename = 'sub_driver.in', subcell = None, params = {}, cell_params = {}, base_in_file = None, **kwargs):
        prefix = os.path.splitext(filename)[0]
        in_params, cards, ase_atoms = self._build_ase_atoms(params, base_in_file, ions = subcell.ions, prefix = prefix)
        self._write_params(filename, ase_atoms, params = in_params, cell_params = cell_params, cards = cards, **kwargs)

    def _fix_params(self, params = None, prefix = 'sub_'):
        default_params = OrderedDict({
                'control' : {
                    'calculation' : 'scf',
                    },
                'system' :
                {
                    'ibrav' : 0,
                    'nosym' : True,
                    },
                'electrons' : {},
                'ions' : {},
                'cell' : {}
                })
        fix_params = {
                'control' :
                {
                    'prefix' : prefix,
                    'outdir' : prefix + '.tmp'  # outdir of qe
                    },
                'electrons' :
                {
                    'conv_thr' : 0.0, # set very high accuracy, otherwise pwscf mixer cause davcio (15) error
                    },
                }
        if not params :
            params = default_params.copy()

        for k1, v1 in fix_params.items() :
            if k1 not in params :
                params[k1] = {}
            for k2, v2 in v1.items() :
                params[k1][k2] = v2
        return params

    def _build_ase_atoms(self, params = None, base_in_file = None, ions = None, prefix = 'sub_'):
        ase_atoms = ions2ase(ions)
        ase_atoms.set_calculator(ase_calc_driver())

        if base_in_file :
            fileobj = open(base_in_file, 'r')
            in_params, card_lines = ase_io_driver.read_fortran_namelist(fileobj)
            fileobj.close()
        else :
            in_params = {}
            card_lines = []

        in_params = self._fix_params(in_params, prefix = prefix)

        for k1, v1 in params.items() :
            if k1 not in in_params :
                in_params[k1] = {}
            for k2, v2 in v1.items() :
                in_params[k1][k2] = v2

        return in_params, card_lines, ase_atoms

    def _write_params(self, outfile, ase_atoms, params = None, cell_params = {}, cards = None, density_initial = None, **kwargs):
        cell_params.update(kwargs)
        if 'pseudopotentials' not in cell_params :
            raise AttributeError("!!!ERROR : Must give the pseudopotentials")
        value = {}
        for k2, v2 in cell_params['pseudopotentials'].items():
            value[k2] = os.path.basename(v2)
        cell_params['pseudopotentials'] = value
        # For QE, all pseudopotentials should at same directory
        params['control']['pseudo_dir'] = os.path.dirname(v2)
        fileobj = open(outfile, 'w')
        if 'kpts' not in cell_params :
            self._update_kpoints(cell_params, cards)
        pw_items = ['control', 'system', 'electrons', 'ions', 'cell']
        pw_params = params.copy()
        for k in params :
            if k not in pw_items : del pw_params[k]
        #-----------------------------------------------------------------------
        rm_items = ['ibrav', 'celldm(1)', 'celldm(2)', 'celldm(3)', 'celldm(4)', 'celldm(5)', 'celldm(6)']
        for item in rm_items :
            if item in pw_params['system'] :
                del pw_params['system'][item]
        #-----------------------------------------------------------------------
        if density_initial and density_initial == 'temp' :
            pw_params['electrons']['startingpot'] = 'file'
        #-----------------------------------------------------------------------
        ase_io_driver.write_espresso_in(fileobj, ase_atoms, pw_params, **cell_params)
        self._write_params_cards(fileobj, params, cards)
        prefix = pw_params['control'].get('prefix', None)
        if 'inputtddft' not in params : params['inputtddft'] = {}
        self._write_params_others(fileobj, params, prefix = prefix)
        fileobj.close()
        return

    def _update_kpoints(self, cell_params, cards = None):
        if cards is None or len(cards) == 0 :
            return
        lines = iter(cards)
        for line in lines :
            if line.split()[0].upper() == 'K_POINTS' :
                ktype = line.split()[1].lower()
                if 'gamma' in ktype :
                    break
                elif 'automatic' in ktype :
                    line = next(lines)
                    item = list(map(int, line.split()))
                    d = {'kpts' : item[:3], 'koffset' : item[3:6]}
                    cell_params.update(d)
                else :
                    raise AttributeError("Not supported by ASE")
                break
        return

    def _write_params_cards(self, fd, params = None, cards = None, **kwargs):
        if cards is None or len(cards) == 0 :
            return
        lines = iter(cards)
        items = ['CONSTRAINTS', 'OCCUPATIONS', 'ATOMIC_FORCES']
        for line in lines :
            if line.split()[0] in items :
                fd.write('\n' + line + '\n')
                for line in lines :
                    if not line[0] == '#' and line.split()[0].isupper():
                        break
                    else :
                        fd.write(line + '\n')
        return

    def _write_params_others(self, fd, params = None, prefix = 'sub_', keys = ['inputtddft'], **kwargs):
        if params is None : return
        fstrl = []
        for section in params:
            if section not in keys : continue
            if section == 'inputtddft' :
                if 'prefix' not in params[section] :
                    params[section]['prefix'] = prefix
                if 'tmp_dir' not in params[section] :
                    params[section]['tmp_dir'] = prefix + '.tmp/'
            fstrl.append('&{0}\n'.format(section.upper()))
            for key, value in params[section].items():
                #-----------------------------------------------------------------------
                if key == 'prefix' :
                    value = prefix
                elif key == 'tmp_dir' :
                    value = prefix + '.tmp/'
                #-----------------------------------------------------------------------
                if value is True:
                    fstrl.append('   {0:40} = .true.\n'.format(key))
                elif value is False:
                    fstrl.append('   {0:40} = .false.\n'.format(key))
                else:
                    fstrl.append('   {0:40} = {1!r:}\n'.format(key, value))
            fstrl.append('/\n\n')
        fd.write(''.join(fstrl))
        return

    @staticmethod
    def get_ions_from_pw():
        alat = qepy.cell_base.get_alat()
        lattice = qepy.cell_base.get_array_at() * alat
        pos = qepy.ions_base.get_array_tau().T * alat
        atm = qepy.ions_base.get_array_atm()
        ityp = qepy.ions_base.get_array_ityp()
        nat = qepy.ions_base.get_nat()
        symbols = [1]
        labels = []
        for i in range(atm.shape[-1]):
            s = atm[:,i].view('S3')[0].strip().decode("utf-8")
            symbols.append(s)

        for i in range(nat):
            labels.append(symbols[ityp[i]])

        ions = Atoms(labels=labels, pos=pos, cell=lattice, basis="Cartesian")
        return ions

    @staticmethod
    def get_density_from_pw(ions, comm = None):
        nr = np.zeros(3, dtype = 'int32')
        qepy.qepy_mod.qepy_get_grid(nr)

        if comm is None or comm.rank == 0 :
            rho = np.zeros((np.prod(nr), 1), order = 'F')
        else :
            rho = np.zeros(3)

        qepy.qepy_mod.qepy_get_rho(rho)

        if comm is None or comm.rank == 0 :
            grid = Grid(ions.pos.cell.lattice, nr)
            density = Field(grid=grid, direct=True, data = rho, order = 'F')
        else :
            density = np.zeros(1)

        return density

    def set_stdout(self, outfile, append = False, **kwargs):
        qepy.qepy_mod.qepy_set_stdout(outfile, append = append)

    def write_stdout(self, line, **kwargs):
        qepy.qepy_mod.qepy_write_stdout(line)
