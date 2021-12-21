#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ase.io
import ase.build
import os
import numpy as np
import argparse

from edftpy.io import ULMReaderMod, ARGDICT, string2array
from edftpy import io
from edftpy.config import read_conf
from edftpy.api.parse_config import config2ions
from edftpy.utils.math import grid_map_data

def get_args():
    parser = argparse.ArgumentParser(description='eDFTpy IO:\n System convert by eDFTpy.\n Atoms is convert by ASE.',
        usage='use "%(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('cells', nargs = '*', help = 'Input structure files(s)\n'
            'If multiple files, the default is perform add. Add the cell for structures, \n'
            'except trajectory file which is append. Add the cell and field for system.')
    parser.add_argument('--convert', dest='convert', action='store_true',
            default=False, help='Convert files formats')
    parser.add_argument('-i', '--ini', '--input', dest='input', type=str, action='store',
            default=None, help='Input structure file')
    # parser.add_argument('-f', '--format', dest='format', type=str, action='store',
    #         default=None, help='The format of input and output')
    # parser.add_argument('-a', '--add', dest='add', action='store_true',
    #         help='Add the cell for structures, except trajectory file which is append. Add the cell and field for system.')
    parser.add_argument('-o', '--output', dest='output', type=str, action='store',
            default=None, help='The output file')
    parser.add_argument('--format-in', dest='format_in', type=str, action='store',
            default=None, help='The format of input')
    parser.add_argument('--format-out', dest='format_out', type=str, action='store',
            default=None, help='The format of output')
    parser.add_argument('-v', '--vacuum', dest='vacuum', type=str, action='store',
            default=None, help='Amount of vacuum to add')
    parser.add_argument('--ext', '--extension', dest='extension', type=str, action='store',
            default=None, help='The extension of output')
    parser.add_argument('-b', '--basefile', dest='basefile', type=str, action='store',
            default=None, help='The additional formation of output.\nIf format_in and format_out are QE/CASTEP input file,\n'
            'also can add other information from the "basefile"')
    parser.add_argument('--additional', dest='additional', type=str, action='store',
            default='', help='The additional part of output')
    parser.add_argument('--scale', dest='scale', type=str, action='store',
            default=None, help='The scale of structure')
    parser.add_argument('--shift', dest='shift', type=str, action='store',
            default=None, help='The shift of structure')
    parser.add_argument('-e', '--elements', dest = "elements", action = ARGDICT,
            default={}, help='Change the type of elements\n    -e "C=O, O = H"')
    parser.add_argument('--subtract', dest='subtract', action='store_true',
            help='Subtract the data from the first one. \nSubtract the same atoms for structures. \nOnly subtract the field for system.')
    parser.add_argument('--frac', '--fraction', dest='frac', action='store_true',
            help='Positions are fractional or not')
    parser.add_argument('--append', dest='append', action='store_true',
            help='Append the stuctures to the first one')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
            help='Output filename same as input')
    parser.add_argument('--traj', dest='traj', action='store_true',
            help='Output the trajectory to one file')
    parser.add_argument('--data-type', dest='data_type', type = str, action='store',
            default = 'density', help='The type of field of system (density/potential). Default is density.')
    parser.add_argument('--json', dest='json', type = str, action='store',
            default =None, help='If the fields are different from each other, \nplease provide the json file (e.g. "edftpy_running.json").')
    parser.add_argument('--qepy', dest='qepy', action='store_true',
            help='From the QE outdir get the density with QEpy. The input file will be the QE input file.')

    args = parser.parse_args()
    return args

def get_outfile(args):
    if args.output is None :
        prefix, ext = os.path.splitext(args.cells[0])
        outf = prefix + '.vasp'
        if args.format_out is not None and args.format_out.lower() in ['eqe', 'qe', 'pwscf', 'espresso-in'] :
            outf = prefix + '.in'
        else :
            ext = args.extension
            if not ext :
                fmt = ase.io.formats.ioformats.get(args.format_out, None)
                ext = fmt.extensions[0] if fmt else 'vasp'
            outf = prefix + '.' + ext
    else :
        outf = args.output
    if args.overwrite : outf = args.cells[0]
    if args.append : outf = args.cells[0]
    #-----------------------------------------------------------------------
    if outf in args.cells[:] and not args.overwrite and not args.append:
        yes=input(f'\033[91m !WARN: \033[00m Are you overwriting the input file {outf}? (yes/no)')
        if yes != 'yes' : exit()
    return outf

def get_atoms(args):
    atoms = None
    for i, fname in enumerate(args.cells) :
        prefix, ext = os.path.splitext(fname)
        format_in = args.format_in
        if format_in is None :
            if ext.lower() == '.in' : format_in = 'espresso-in'
        if args.format_in == 'traj' or ext.lower() == '.traj' :
            from ase.io.trajectory import Trajectory
            struct = Trajectory(fname)
            # Try to fix the trajectory file
            if len(struct) == 0 :
                struct.close()
                struct = ULMReaderMod(fname)
                if len(struct)>0 :
                    print(f'\033[92m !OK: \033[00m Fixed the trajectory file {fname}')
                else :
                    raise AttributeError(f'!ERROR : The trajectory file {fname} has a big problem that cannot fix.')
            if not args.traj :
                struct = struct[-1]
        elif ext.lower() == '.json' :
            config = read_conf(fname)
            config["GSYSTEM"]["cell"]["file"] = ''
            ions = config2ions(config)
            struct = io.ions2ase(ions)
        else :
            try:
                struct = ase.io.read(fname, format=format_in)
            except Exception :
                try:
                    struct = io.read(fname, format=format_in)
                    struct = io.ions2ase(struct)
                except Exception as e:
                    raise e

        if args.traj :
            append = False if i == 0 else True
            if i == 0 and args.append : continue
            ase.io.write(args.output, struct, format = args.format_out, append = append)
        else :
            if i>0 :
                if args.subtract :
                    atoms = io.ase_io.subtract_atoms(atoms, struct)
                else :
                    atoms += struct
            else :
                atoms = struct
    return atoms

def get_system(args):
    system = None
    iolist = ['snpy', 'xsf', 'pp', 'qepp']
    has_field = True
    for fname in (*args.cells, args.output) :
        prefix, ext = os.path.splitext(fname)
        if ext.lower()[1:] not in iolist :
            has_field = False
            return None

    if has_field :
        lens = max([len(item) for item in args.cells] + [len(args.output)])
        for i, fname in enumerate(args.cells) :
            struct=io.read_system(fname, data_type = args.data_type)
            nele = struct.field.integral()
            print(f'Number_of_electrons {fname:{lens}s} : {nele}', flush = True)
            if i>0 :
                if not np.all(system.field.shape == struct.field.shape):
                    if len(args.cells) == 2 :
                        # print(f'!WARN: We interpolate the sencond {struct.field.shape} to the first one {system.field.shape}. If sDFT, please with json file.', flush = True)
                        print(f'\033[91m !WARN: \033[00m We interpolate the second {struct.field.shape} to the first one {system.field.shape}. If sDFT, please with json file.', flush = True)
                        struct.field = grid_map_data(struct.field, grid = system.field.grid)
                    else :
                        raise AttributeError("The shapes are not matched, If sDFT, try with json file?")
                if args.subtract :
                    system.field -= struct.field
                else :
                    system += struct
            else :
                system = struct
        return (system.ions, system.field)

def get_system_json(args):
    from edftpy.utils.common import Field, Grid
    from edftpy.mpi.mpi import Graph
    config = read_conf(args.json)
    config["GSYSTEM"]["cell"]["file"] = ''
    gions = config2ions(config)
    cell = gions.pos.cell
    # lattice = config["GSYSTEM"]['cell']['lattice']
    nr = config["GSYSTEM"]['grid']['nr']
    grid = Grid(lattice=cell, nr=nr)
    spacings = grid.spacings
    field = Field(grid)
    ions = None
    graph = Graph(grid = grid)
    lens = max([len(item) for item in args.cells] + [len(args.output)])
    for i, fname in enumerate(args.cells) :
        prefix, _ = os.path.splitext(fname)
        prefix = prefix.upper()
        if prefix.startswith('SUB'):
            if prefix not in config :
                raise AttributeError(f"The {prefix} not in the json file.")
            snr = config[prefix]['grid']['nr']
            shift = config[prefix]['grid']['shift']
            graph.sub_shift[0] = shift
            graph.sub_shape[0] = snr
        else :
            shift = np.zeros(3, dtype = 'int32')
        struct=io.read_system(fname, data_type = args.data_type)
        nele = struct.field.integral()
        print(f'Number_of_electrons {fname:{lens}s} : {nele}', flush = True)
        if i>0 :
            if prefix.startswith('SUB'):
                index = graph.get_sub_index(0, in_global = True)
            else :
                index = slice(None)
            if args.subtract :
                field[index] -= struct.field
            else :
                ions = ions + struct.ions.translate(shift*spacings)
                field[index] += struct.field
        else :
            ions = struct.ions.translate(shift*spacings)
            ions.set_cell(cell)
            if prefix.startswith('SUB'):
                index = graph.get_sub_index(0, in_global = True)
            else :
                index = slice(None)
            field[index] += struct.field
    return (ions, field)

def get_qepy_system(args):
    from edftpy.engine.engine_qe import EngineQE
    import ase.io.espresso as ase_io_qe
    import qepy
    inputobj = qepy.qepy_common.input_base()
    #-----------------------------------------------------------------------
    if len(args.cells)>1 :
        raise AttributeError("Sorry, each time only can convert one file by QEpy.")
    fname = args.cells[0]
    with open(fname, 'r') as fh:
        params, card_lines = ase_io_qe.read_fortran_namelist(fh)
    prefix = params.get('control', {}).get('prefix', 'pwscf').strip()
    tmp_dir= params.get('control', {}).get('outdir', '.').strip() + os.sep
    #-----------------------------------------------------------------------
    data_dir = tmp_dir + prefix + '.save' + os.sep
    if os.path.isfile(data_dir + 'data-file-schema.xml') :
        oldxml = False
    elif os.path.isfile(data_dir + 'data-file.xml') :
        oldxml = True
    else : # try eqe outdir
        fpref, _ = os.path.splitext(fname)
        num = fpref.split('_')[-1]
        tmp_dir0 = tmp_dir
        if tmp_dir.strip('./') :
            tmp_dir = tmp_dir.strip('/').strip(os.sep) +'_' + num + os.sep
        else :
            tmp_dir = 'tmp_' + num + os.sep
        data_dir = tmp_dir + prefix + '.save' + os.sep
        if os.path.isfile(data_dir + 'data-file.xml') :
            oldxml = True
        else :
            raise AttributeError(f"Can not find the 'xml' file in any tmp_dir({tmp_dir0} or {tmp_dir}).")
    inputobj.prefix = prefix
    inputobj.tmp_dir= tmp_dir
    #-----------------------------------------------------------------------
    qepy.qepy_initial(inputobj)
    if oldxml :
        if not hasattr(qepy, 'oldxml_read_file'):
            raise AttributeError("Please reinstall the QEpy with 'oldxml=yes'.")
        qepy.oldxml_read_file()
    else :
        qepy.read_file()
    ions = EngineQE.get_ions_from_pw()
    rho = EngineQE.get_density_from_pw(ions)

    qepy.qepy_stop_run(0, what = 'no')
    return(ions, rho)

def run(args):
    if len(args.cells) == 0 :
        args.cells.append(args.input)
    if len(args.cells) == 0 :
        raise AttributeError("Provide at least one file")
    #-----------------------------------------------------------------------
    if args.format_in is not None and args.format_in.lower() in ['eqe', 'qe', 'pwscf', 'espresso-in'] :
        args.format_in = 'espresso-in'
    #-----------------------------------------------------------------------
    args.output = get_outfile(args)
    if args.json :
        system = get_system_json(args)
    elif args.qepy :
        system = get_qepy_system(args)
    else :
        system = get_system(args)
    if system is not None :
        lens = max([len(item) for item in args.cells] + [len(args.output)])
        if args.subtract :
            nele = np.abs(system[1]).integral()*0.5
            print(f'Differ_of_electrons {args.output:{lens}s} : {nele}')
        else :
            nele = system[1].integral()
            print(f'Number_of_electrons {args.output:{lens}s} : {nele}')
        io.write(args.output, ions = system[0], data = system[1], format = args.format_out, data_type = args.data_type)
        return
    atoms = get_atoms(args)
    if args.traj : return
    #-----------------------------------------------------------------------
    if args.scale:
        atoms.set_cell(atoms.get_cell()*string2array(args.scale), scale_atoms=True)

    if args.shift:
        atoms.translate(string2array(args.shift))

    if args.vacuum :
        array = string2array(args.vacuum)
        if len(array) == 1 :
            vacuum = array
            axis = (0, 1, 2)
        else :
            vacuum = np.max(array)
            axis = np.where(array > 0)[0]

        atoms.center(vacuum = vacuum, axis = axis)
        atoms.set_pbc(True)

    if len(args.elements) > 0 :
        symbols = atoms.get_chemical_symbols()
        for i, item in enumerate(symbols):
            for k, v in args.elements.items() :
                if item == k :
                    symbols[i] = v
        atoms.set_chemical_symbols(symbols)

    if args.output.lower().endswith(('vasp', 'poscar', 'contcar')):
        atoms = io.ase_io.sort_ase_atoms(atoms)

    kwargs = {}

    fout = args.format_out
    if not fout :
        fout = os.path.splitext(args.output)[1][1:]
        if not fout :
            fout = os.path.basename(args.output)
        fout = fout.lower()
    if fout.startswith('castep'):
        if args.frac : kwargs['positions_frac'] = True
    if fout in ['vasp', 'poscar', 'contcar'] :
        kwargs['vasp5'] = True
        if args.frac : kwargs['direct'] = True
    if fout in ['eqe', 'qe', 'pwscf', 'espresso-in'] :
        if args.frac : kwargs['crystal_coordinates'] = True

    if fout in ['eqe', 'qe', 'pwscf', 'espresso-in'] :
        io.ase_write2qe(args.output, atoms, args.basefile, **kwargs)
    elif fout in ['castep', 'castep-cell'] :
        io.ase_write2castep(args.output, atoms, args.basefile, **kwargs)
    else :
        ase.io.write(args.output, atoms, format = args.format_out, **kwargs)

    if len(args.additional) > 0 :
        fstr = str(args.additional)
        fstr = fstr.replace('\\n', '\n')
        with open(args.output, 'a+') as fw:
            fw.write(fstr)

def main():
    args = get_args()
    return run(args)


if __name__ == "__main__":
    main()
