import qepy
import os
import ase.io.espresso as ase_io_driver
from ase.calculators.espresso import Espresso as ase_calc_driver
from collections import OrderedDict

from dftpy.constants import LEN_CONV

from edftpy.io import ions2ase
from edftpy.engine.driver import Engine


class EngineQE(Engine):
    def __init__(self, **kwargs):
        unit_len = LEN_CONV["Bohr"]["Angstrom"] / qepy.constants.BOHR_RADIUS_SI / 1E10
        unit_vol = unit_len ** 3
        units = kwargs.get('units', {})
        kwargs['units'] = units
        kwargs['units']['volume'] = unit_vol
        super().__init__(**kwargs)

    def get_force(self, icalc = 0, **kwargs):
        qepy.qepy_forces(icalc)
        forces = qepy.force_mod.get_array_force().T / 2.0  # Ry to a.u.
        return forces

    def embed_base(self, exttype = 0, diag_conv = 1E-6, **kwargs):
        embed = qepy.qepy_common.embed_base()
        embed.exttype = exttype
        embed.diag_conv = diag_conv
        return embed

    def calc_energy(self, embed, **kwargs):
        qepy.qepy_calc_energies(embed)
        energy = embed.etotal * 0.5 # Ry to a.u.
        return energy

    def clean_saved(self, *args, **kwargs):
        qepy.qepy_clean_saved()

    def get_grid(self, nr, **kwargs):
        qepy.qepy_mod.qepy_get_grid(nr)

    def get_rho(self, rho, **kwargs):
        qepy.qepy_mod.qepy_get_rho(rho)

    def get_rho_core(self, rho, **kwargs):
        qepy.qepy_mod.qepy_get_rho_core(rho)

    def get_ef(self, **kwargs):
        return qepy.ener.get_ef()

    def initial(self, inputfile, comm = None, **kwargs):
        qepy.qepy_pwscf(inputfile, comm)

    def save(self, save = ['D'], **kwargs):
        if 'W' in save :
            what = 'all'
        else :
            what = 'config-nowf'
        qepy.punch(what)
        qepy.close_files(False)

    def scf(self, embed, initial = True, **kwargs):
        embed.mix_coef = -1.0
        embed.finish = False
        embed.initial = initial
        qepy.qepy_electrons_scf(0, 0, embed)

    def scf_mix(self, embed, coef = 0.7, **kwargs):
        embed.mix_coef = coef
        qepy.qepy_electrons_scf(2, 0, embed)

    def set_extpot(self, embed, extpot, **kwargs):
        extpot = extpot * 2.0 # a.u. to Ry
        qepy.qepy_mod.qepy_set_extpot(embed, extpot)

    def set_rho(self, rho, **kwargs):
        qepy.qepy_mod.qepy_set_rho(rho)

    def set_stdout(self, outfile, append = False, **kwargs):
        qepy.qepy_mod.qepy_set_stdout(outfile, append = append)

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

    def end_scf(self, embed, **kwargs):
        embed.finish = True
        qepy.qepy_electrons_scf(0, 0, embed)

    def end_tddft(self, embed, **kwargs):
        embed.tddft.finish = True
        qepy.qepy_molecule_optical_absorption(embed)

    def tddft(self, embed, **kwargs):
        qepy.qepy_molecule_optical_absorption(embed)

    def tddft_after_scf(self, inputfile, embed, **kwargs):
        embed.tddft.initial = True
        embed.tddft.finish = False
        embed.tddft.nstep = 900000 # Any large enough number
        qepy.qepy_tddft_readin(inputfile)
        qepy.qepy_tddft_main_setup(embed)

    def tddft_initial(self, inputfile, comm = None, **kwargs):
        qepy.qepy_tddft_main_initial(inputfile, comm)
        qepy.read_file()

    def update_ions(self, embed, subcell, update = 0, **kwargs):
        pos = subcell.ions.pos.to_cart().T / subcell.grid.latparas[0]
        qepy.qepy_mod.qepy_update_ions(embed, pos, update)

    def wfc2rho(self, *args, **kwargs):
        qepy.qepy_tddft_mod.qepy_cetddft_wfc2rho()

    def write_stdout(self, line, **kwargs):
        qepy.qepy_mod.qepy_write_stdout(line)

    def write_input(self, filename = 'sub_driver.in', subcell = None, params = {}, cell_params = {}, base_in_file = None, **kwargs):
        prefix = os.path.splitext(filename)[0]
        in_params, cards, ase_atoms = self._build_ase_atoms(params, base_in_file, ions = subcell.ions, prefix = prefix)
        self._write_params(filename, ase_atoms, params = in_params, cell_params = cell_params, cards = cards)

    def _fix_params(self, params = None, prefix = 'sub_'):
        #output of qe
        default_params = OrderedDict({
                'control' :
                {
                    'calculation' : 'scf',
                    # 'verbosity' : 'high',
                    'restart_mode' : 'from_scratch',
                    # 'iprint' : 1,
                    'disk_io' : 'none', # do not save anything for qe
                    },
                'system' :
                {
                    'ibrav' : 0,
                    'nat' : 1,
                    'ntyp' : 1,
                    # 'ecutwfc' : 40,
                    # 'nosym' : True,
                    'occupations' : 'smearing',
                    'degauss' : 0.001,
                    'smearing' : 'gaussian',
                    },
                'electrons' :
                {
                    'mixing_beta' : 0.5,
                    },
                'ions' :
                {
                    'pot_extrapolation' : 'atomic',  # extrapolation for potential from preceding ionic steps
                    'wfc_extrapolation' : 'none'   # no extrapolation for wfc from preceding ionic steps
                    },
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
                    'electron_maxstep' : 1, # only run 1 step for scf
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

    def _write_params(self, outfile, ase_atoms, params = None, cell_params = {}, cards = None, **kwargs):
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
        ase_io_driver.write_espresso_in(fileobj, ase_atoms, pw_params, **cell_params)
        self._write_params_cards(fileobj, params, cards)
        prefix = pw_params['control'].get('prefix', None)
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

    def _write_params_others(self, fd, params = None, prefix = 'sub_', **kwargs):
        if params is None or len(params) == 0 :
            return
        fstrl = []
        # pw_keys = ['control', 'system', 'electrons', 'ions', 'cell']
        keys = ['inputtddft']
        for section in params:
            if section not in keys : continue
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

