import qepy
from qepy.driver import Driver
import numpy as np
import os
import ase.io.espresso as ase_io_driver
from ase.calculators.espresso import Espresso as ase_calc_driver
from collections import OrderedDict

from dftpy.constants import LEN_CONV

from edftpy.io import ions2ase
from edftpy.engine.engine import Engine

try:
    __version__ = qepy.__version__
except Exception :
    __version__ = '0.0.1'

class EngineQE(Engine):
    def __init__(self, nscf = False, **kwargs):
        unit_len = LEN_CONV["Angstrom"]["Bohr"] / qepy.constants.ANGSTROM_AU
        units = kwargs.get('units', {})
        units['length'] = kwargs.get('length', unit_len)
        units['energy'] = kwargs.get('energy', 0.5)
        units['order'] = 'F'
        kwargs['units'] = units
        super().__init__(**kwargs)
        self.embed = None
        self.nscf = nscf
        self.driver = Driver

    def get_forces(self, icalc = 3, **kwargs):
        return self.driver.get_forces(icalc=icalc, **kwargs)

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
        return self.get_energy(self, **kwargs)

    def get_energy(self, olevel = 0, **kwargs):
        if olevel == 0 : self.driver.calc_energy(**kwargs)
        return self.embed.etotal

    def get_grid(self, **kwargs):
        return self.driver.get_number_of_grid_points()

    def get_rho(self, rho, **kwargs):
        return self.driver.get_density(out=rho)

    def get_rho_core(self, rho, **kwargs):
        return self.driver.get_core_density(out=rho)

    def get_ef(self, **kwargs):
        return self.driver.get_fermi_level()

    def _initial_files(self, inputfile = None, comm = None, **kwargs):
        self.comm = comm or self.comm

        if inputfile is None :
            prefix = kwargs.get('prefix', 'sub_qe')
            inputfile = prefix + '.in'
            if self.comm.rank == 0 :
                self.write_input(inputfile, **kwargs)
            if self.comm.size > 1 : self.comm.Barrier()
        self.inputfile = inputfile

    def initial(self, inputfile = None, comm = None, task = 'scf', **kwargs):
        self._initial_files(inputfile = inputfile, comm = comm, **kwargs)
        self.embed = self.embed_base(**kwargs)
        self.driver = Driver(inputfile = self.inputfile, comm = self.comm, task = task,
                embed = self.embed, iterative = self.embed.iterative, **kwargs)

    def tddft_initial(self, inputfile = None, comm = None, task = 'optical', **kwargs):
        self.initial(inputfile=inputfile, comm=comm, task=task, **kwargs)

    def save(self, save = ['D'], **kwargs):
        if 'W' in save :
            what = 'all'
        else :
            what = 'config-nowf'
        self.driver.save(what = what, **kwargs)

    def scf(self, sum_band = True, **kwargs):
        self.driver.diagonalize(nscf = self.nscf, **kwargs)
        if self.nscf :
            if sum_band : self.sum_band()

    def sum_band(self, occupations= None, **kwargs):
        if occupations is not None :
            nbnd = self.driver.get_number_of_bands()
            nk = self.driver.get_number_of_k_points()
            wg = np.zeros((nbnd, nk))
            occupations = occupations.ravel().reshape((-1, nk), order='F')
            nbands = min(wg.shape[0], occupations.shape[0])
            wg[:nbands] = occupations[:nbands]
            occupations = wg
        self.driver.sum_band(occupations = occupations)

    def get_band_energies(self, **kwargs):
        return self.driver.get_eigenvalues(kpt=slice(None))

    def get_band_weights(self, **kwargs):
        wk = self.driver.get_k_point_weights()
        nbands = self.driver.get_number_of_bands()
        weights = np.repeat(wk, nbands).reshape((-1, nbands))
        return weights

    def scf_mix(self, coef = 0.7, **kwargs):
        self.driver.mix(mix_coef = coef, **kwargs)

    def set_extpot(self, extpot, **kwargs):
        self.driver.set_external_potential(extpot)

    def set_rho(self, rho, **kwargs):
        self.driver.set_density(rho, **kwargs)

    def stop_scf(self, status = 0, save = ['D'], **kwargs):
        if 'W' in save :
            what = 'all'
        else :
            what = 'config-nowf'
        self.driver.stop(exit_status = status, what = what, **kwargs)

    def stop_tddft(self, status = 0, save = ['D'], **kwargs):
        self.stop_scf(status, save, **kwargs)

    def end_scf(self, **kwargs):
        self.driver.end_scf(**kwargs)

    def end_tddft(self, **kwargs):
        self.end_scf(**kwargs)

    def tddft(self, **kwargs):
        self.driver.diagonalize(**kwargs)

    def tddft_after_scf(self, inputfile = None, **kwargs):
        inputfile = inputfile or self.inputfile
        self.driver = Driver(inputfile = inputfile, comm = self.comm, task = 'optical',
                embed = self.embed, iterative = self.embed.iterative, progress = True, **kwargs)

    def tddft_restart(self, istep=None, **kwargs):
        self.driver.tddft_restart(istep=istep, **kwargs)

    def update_ions(self, subcell, update = 0, **kwargs):
        self.driver.update_ions(positions = subcell.ions.positions, update = 0, **kwargs)

    def get_dnorm(self, **kwargs):
        return self.embed.dnorm

    def set_dnorm(self, dnorm, **kwargs):
        self.embed.dnorm = dnorm

    def check_convergence(self, **kwargs):
        return self.driver.check_convergence(**kwargs)

    def write_input(self, filename = 'sub_driver.in', subcell = None, params = {}, cell_params = {}, base_in_file = None, **kwargs):
        prefix = os.path.splitext(filename)[0]
        in_params, cell_params, cards, ase_atoms = self._build_ase_atoms(params = params,
                base_in_file = base_in_file, cell_params = cell_params, ions = getattr(subcell, 'ions', None), prefix = prefix)
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

    def _check_params(self, params = None, cell_params = None, card_lines = None, ase_atoms = None):
        if params['system'].get('nspin', 1) > 1 :
            if 'tot_magnetization' not in params['system'] :
                for key in params['system'].keys():
                    if key.startswith('starting_magnetization'): break
                else :
                    params['system']['tot_magnetization'] = 0
        return params

    def _build_ase_atoms(self, params = {}, cell_params = {}, base_in_file = None, ions = None, prefix = 'sub_'):
        in_params = {}
        card_lines = []

        if ions is not None :
            ase_atoms = ions2ase(ions)
        else :
            ase_atoms = None

        if base_in_file :
            if not isinstance(base_in_file, (list, tuple)): base_in_file = [base_in_file]
            in_params, in_cell_params, ase_atoms = self.merge_inputs(*base_in_file, atoms = ase_atoms,
                    pseudopotentials = cell_params.get('pseudopotentials', None))
            if cell_params : in_cell_params.update(cell_params)
            cell_params = in_cell_params
            if len(base_in_file) == 1 :
                with open(base_in_file[0], 'r') as fh:
                    _, card_lines = ase_io_driver.read_fortran_namelist(fh)

        ase_atoms.set_calculator(ase_calc_driver())

        if params :
            in_params = self._fix_params(in_params, prefix = prefix)
            for k1, v1 in params.items() :
                if k1 not in in_params :
                    in_params[k1] = {}
                for k2, v2 in v1.items() :
                    in_params[k1][k2] = v2

        self._check_params(params=in_params, cell_params=cell_params, card_lines=card_lines, ase_atoms=ase_atoms)
        return in_params, cell_params, card_lines, ase_atoms

    def _write_params(self, outfile, ase_atoms, params = None, cell_params = {}, cards = None, density_initial = None, **kwargs):
        cell_params.update(kwargs)
        if 'pseudopotentials' not in cell_params :
            raise AttributeError("!!!ERROR : Must give the pseudopotentials")
        value = OrderedDict()
        for k2, v2 in cell_params['pseudopotentials'].items():
            value[k2] = os.path.basename(v2)
        cell_params['pseudopotentials'] = value
        # For QE, all pseudopotentials should at same directory
        params['control']['pseudo_dir'] = os.path.dirname(v2) or './'
        if 'kpts' not in cell_params :
            self._update_kpoints(cell_params, cards)
        kpts = cell_params.get('kpts', [0])
        if kpts[0] == 0 : cell_params['kpts'] = None
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
        fileobj = open(outfile, 'w')
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
    def merge_inputs(*args, atoms = None, pseudopotentials = None, **kwargs):
        """
        Multiple input files into one file.
        Note :
            Please check the results carefully.
        """
        import re
        inputs = OrderedDict({
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
        qe_kws_dims = {'system' :[
                'starting_charge',
                'starting_magnetization',
                'hubbard_u',
                'hubbard_j0',
                'hubbard_alpha',
                'hubbard_beta',
                'angle1',
                'angle2',
                'london_c6',
                'london_rvdw',
                'starting_ns_eigenvalue',
                ]}
        qe_kws_number= {
                'control' : {
                    'nstep': [int, max],
                    'iprint': [int, max],
                    'dt': [float, max],
                    'max_seconds': [float, max],
                    'etot_conv_thr': [float, max],
                    'forc_conv_thr': [float, max],
                    'nberrycyc': [int, max],
                    'gdir': [int, max],
                    'nppstr': [int, max],
                    },
                'system' : {
                    'ibrav': [int, min],
                    'nat': [int, sum],
                    'ntyp': [int, max],
                    'nbnd': [int, sum],
                    'tot_charge': [float, sum],
                    'tot_magnetization': [float, sum],
                    'ecutwfc': [float, max],
                    'ecutrho': [float, max],
                    'ecutfock': [float, max],
                    'degauss': [float, max],
                    'nspin': [int, max],
                    'ecfixed': [float, max],
                    'qcutz': [float, max],
                    'q2sigma': [float, max],
                    'exx_fraction': [float, max],
                    'screening_parameter': [float, max],
                    'ecutvcut': [float, max],
                    'localization_thr': [float, max],
                    'lda_plus_u_kind': [int, max],
                    'edir': [int, max],
                    'emaxpos': [float, max],
                    'eopreg': [float, max],
                    'eamp': [float, max],
                    'fixed_magnetization(1)': [float, sum],
                    'fixed_magnetization(2)': [float, sum],
                    'fixed_magnetization(3)': [float, sum],
                    'lambda': [float, max],
                    'report': [int, max],
                    'esm_w': [float, max],
                    'esm_efield': [float, max],
                    'esm_nfit': [int, max],
                    'fcp_mu': [float, max],
                    'london_s6': [float, max],
                    'london_rcut': [float, max],
                    'ts_vdw_econv_thr': [float, max],
                    'xdm_a1': [float, max],
                    'xdm_a2': [float, max],
                    'space_group': [int, min],
                    'origin_choice': [int, max],
                    'zgate': [float, max],
                    'block_1': [float, max],
                    'block_2': [float, max],
                    'block_height': [float, max],
                    },
                'electrons' : {
                        'electron_maxstep': [int, max],
                        'conv_thr': [float, max],
                        'conv_thr_init': [float, max],
                        'conv_thr_multi': [float, max],
                        'mixing_beta': [float, min],
                        'mixing_ndim': [int, max],
                        'mixing_fixed_ns': [int, max],
                        'ortho_para': [int, max],
                        'diago_thr_init': [float, max],
                        'diago_cg_maxiter': [int, max],
                        'diago_david_ndim': [int, max],
                        'efield': [float, max],
                        'efield_cart(1)': [float, max],
                        'efield_cart(2)': [float, max],
                        'efield_cart(3)': [float, max],
                        }}
        #-----------------------------------------------------------------------
        if atoms is None :
            for fname in args:
                with open(fname) as fh:
                    if atoms is None :
                        atoms = ase_io_driver.read_espresso_in(fh)
                    else :
                        atoms = atoms + ase_io_driver.read_espresso_in(fh)
        pattern=re.compile(r'(.*?)(\d+)\)')
        #-----------------------------------------------------------------------
        in_params_all  = []
        atomic_species_all = []
        k_points_all = []
        pps = {}
        for i, fname in enumerate(args):
            fileobj = open(fname, 'r')
            in_params, card_lines = ase_io_driver.read_fortran_namelist(fileobj)
            fileobj.close()
            #-----------------------------------------------------------------------
            atomic_species = []
            k_points = {}
            lines = iter(card_lines)
            for line in lines :
                if line.split()[0].upper() == 'K_POINTS' :
                    ktype = line.split()[1].lower()
                    if 'gamma' in ktype :
                        # k_points = {'kpts' : [0]*3, 'koffset' : [0]*3}
                        k_points = {}
                    elif 'automatic' in ktype :
                        line = next(lines)
                        item = list(map(int, line.split()))
                        k_points = {'kpts' : item[:3], 'koffset' : item[3:6]}
                elif line.split()[0].upper() == 'ATOMIC_SPECIES' :
                    ntyp = int(in_params['system']['ntyp'])
                    for i in range(ntyp):
                        line = next(lines)
                        symbol, _ , pp = line.split()[:3]
                        atomic_species.append(symbol.capitalize())
                        pps[atomic_species[-1]] = pp
            in_params_all.append(in_params)
            atomic_species_all.append(atomic_species)
            k_points_all.append(k_points)
            #-----------------------------------------------------------------------
            for section in in_params:
                if section in inputs :
                    inputs[section].update(in_params[section])
                else :
                    inputs[section] = in_params[section].copy()
        #-----------------------------------------------------------------------
        if pseudopotentials : pps = pseudopotentials
        pseudopotentials = {}
        symbols = OrderedDict.fromkeys(atoms.get_chemical_symbols())
        for symbol in symbols :
            if symbol not in pseudopotentials :
                pseudopotentials[symbol] = pps.get(symbol, None)
        ppkeys = list(pseudopotentials.keys())
        #-----------------------------------------------------------------------
        # remove all dimensions keys
        for section in list(qe_kws_dims.keys()):
            dims_keys = []
            keys = list(inputs[section].keys())
            for key in keys:
                for pref in qe_kws_dims[section] :
                    if key.startswith(pref):
                        # dims_keys.append(pref)
                        dims_keys.append(key)
                        inputs[section].pop(key)
            qe_kws_dims[section] = dims_keys

        # update all dimensions keys
        for section, item in qe_kws_dims.items() :
            for i, in_params in enumerate(in_params_all):
                atomic_species = atomic_species_all[i]
                if len(atomic_species) == 0 : continue
                for keyd in item :
                    if keyd in in_params[section] :
                        m = pattern.match(keyd)
                        s = atomic_species[int(m.group(2)) - 1]
                        if s not in symbols : continue
                        ind = ppkeys.index(s) + 1
                        newkeyd = m.group(1)+str(ind) + ')'
                        inputs[section][newkeyd] = in_params[section][keyd]

        # update number keys
        for section, item in qe_kws_number.items() :
            for key, item2 in item.items():
                l = []
                for in_params in in_params_all:
                    if key in in_params.get(section, {}):
                        l.append(item2[0](in_params[section][key]))
                if len(l)>0 : inputs[section][key] = item2[1](l)

        kpts = np.zeros(3, dtype = np.int64)
        koffset = np.zeros(3, dtype = np.int64)
        for k_points in k_points_all :
            if 'kpts' in k_points :
                kpts = np.maximum(kpts, k_points['kpts'])
                koffset = k_points.get('koffset', koffset)

        #-----------------------------------------------------------------------
        inputs['system']['ntyp'] = len(pseudopotentials)
        if 'lda_plus_u' in inputs['system'] :
            lda_plus_u = False
            for key in inputs['system'] :
                if key.lower().startswith('hubbard'):
                    lda_plus_u = True
                    break
            inputs['system']['lda_plus_u'] = lda_plus_u
        #-----------------------------------------------------------------------

        cell_params = {
                'pseudopotentials' : pseudopotentials,
                'kpts' : kpts,
                'koffset' : koffset,
                }
        return inputs, cell_params, atoms
