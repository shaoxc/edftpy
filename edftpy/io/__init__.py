import os
from functools import wraps
import numpy as np
import argparse
from pathlib import Path
import ase.io
from ase.io import ulm

from dftpy.formats import ase_io
from dftpy.formats.io import read, read_all, read_density, read_potential, \
        write, write_all, write_density, write_potential, \
        guess_format
from dftpy.formats.ase_io import ase_read, ase_write, ase2ions, ions2ase

def print2file(fileobj = None):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if fileobj is None :
                fobj = args[0].fileobj
            elif isinstance(fileobj, str):
                fobj = open(fileobj, 'a')
            else :
                fobj = fileobj
            stdout = None
            if fobj is not None :
                if os.fstat(1).st_ino != os.fstat(fobj.fileno()).st_ino :
                    stdout = os.dup(1)
                    os.dup2(fobj.fileno(), 1)
            results = function(*args, **kwargs)
            if stdout is not None :
                os.dup2(stdout, 1)
                os.close(stdout)
                if isinstance(fileobj, str): fobj.close()
            return results
        return wrapper
    return decorator

def string2array(fstr, dtype = 'float', sep = None):
    string = fstr.lower().replace('d', 'e')
    if sep is None :
        if ',' in string[:1000] : # For speed, only find sep in 1000 chars.
            sep = ','
        else :
            sep = ' '
    array = np.fromstring(string.strip(), dtype=dtype, sep=sep)
    return array

class ARGDICT(argparse.Action) :
    def __call__(self , parser, namespace, values, option_string = None):
        setattr(namespace, self.dest, {})

        for item in values.split(',') :
            key, value = item.split('=')
            key = key.strip()
            value = value.strip()
            getattr(namespace, self.dest)[key] = value

class ULMReaderMod(ulm.Reader):
    def __init__(self, fd, index=0, data=None, _little_endian=None):
        self._little_endian = _little_endian
        if not hasattr(fd, 'read'): fd = Path(fd).open('r+b')
        self._fd = fd
        self._index = index
        self._tag, self._version, self._nitems, self._pos0, self._offsets =ulm.read_header(fd)
        if self._nitems == 0 :
            fp = fd.tell()
            for n in range(10000000):
                offsets = ulm.readints(fd, n)
                fd.seek(fp)
                if len(offsets)>1 and offsets[-1] < offsets[-2] :
                    offsets = ulm.readints(fd, n-1)
                    break
            self._nitems = n - 1
            self._offsets = offsets
            ulm.writeint(fd, self._nitems, 32)
        data = self._read_data(index)
        self._parse_data(data)

def ase_write2qe(outf, atoms, basefile = None, crystal_coordinates = False, **kwargs):
    if basefile is None :
        ase.io.write(outf, atoms, format = 'espresso-in', crystal_coordinates = crystal_coordinates, **kwargs)
    else :
        symbols = set(atoms.symbols)
        ntyp = len(symbols)
        nat = atoms.get_global_number_of_atoms()
        with open(basefile, 'r') as fr :
            with open(outf, 'w') as fw :
                for line in fr :
                    if 'ntyp' in line :
                        ntyp_old = int(line.split('=')[1])
                        x = line.index("=") + 1
                        line = line[:x] + ' ' + str(ntyp) + '\n'
                    elif 'atomic_species' in line.lower() :
                        for i in range(ntyp_old):
                            pp = fr.readline()
                            s = pp.split()[0].capitalize()
                            if s in symbols :
                                line += pp
                    elif 'nat' in line :
                        nat_old = int(line.split('=')[1])
                        x = line.index("=") + 1
                        line = line[:x] + ' ' + str(nat) + '\n'
                    elif 'cell_parameters' in line.lower() :
                        for i in range(3):
                            fr.readline()
                            line += '{0[0]:.14f} {0[1]:.14f} {0[2]:.14f}\n'.format(atoms.cell[i])
                    elif 'atomic_positions' in line.lower():
                        for i in range(nat_old):
                            fr.readline()
                        if crystal_coordinates:
                            line = 'ATOMIC_POSITIONS crystal\n'
                            positions = atoms.positions
                        else:
                            line = 'ATOMIC_POSITIONS angstrom\n'
                            positions = atoms.get_scaled_positions()
                        for s, p in zip(atoms.symbols, positions):
                            line += '{0:4s} {1[0]:.14f} {1[1]:.14f} {1[2]:.14f}\n'.format(s, p)
                    fw.write(line)

def ase_write2castep(outf, atoms, basefile = None, **kwargs):
    ase.io.write(outf, atoms, format = 'castep-cell', **kwargs)
    symbols = set(atoms.get_chemical_symbols())
    prev = 'LAST'
    if basefile is not None :
        with open(basefile, 'r') as fr :
            with open(outf, 'a') as fw :
                for line in fr :
                    line = line.lstrip()
                    if line.startswith('#') : continue
                    if line.lower().startswith('%block lattice'):
                        for line in fr :
                            line = line.strip()
                            if line.lower().startswith('%endblock lattice'): break
                        line = fr.readline()
                    if line.lower().startswith('%block position'):
                        for line in fr :
                            line = line.strip()
                            if line.lower().startswith('%endblock position'): break
                        line = fr.readline()
                    if line.lower().startswith('%block species_pot'):
                        pots = {}
                        for line in fr :
                            line = line.strip()
                            if line.lower().startswith('%endblock species_pot'):
                                break
                            else :
                                s, p = line.split()
                                pots[s] = p
                        line = '%BLOCK SPECIES_POT\n'
                        for s in symbols :
                            if s in pots :
                                line += "{:<4s} {}\n".format(s, pots[s])
                            else :
                                raise AttributeError("Missing potential for {}".format(s))
                        line += '%ENDBLOCK SPECIES_POT\n\n'

                    if len(prev.strip())== len(line.strip())== 0 : continue
                    fw.write(line)
                    line, prev = prev, line
