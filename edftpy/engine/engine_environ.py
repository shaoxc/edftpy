import sys

import pyec
import pyec.setup_interface as environ_setup
import pyec.control_interface as environ_control
import pyec.calc_interface as environ_calc
import pyec.output_interface as environ_output

import numpy as np

from dftpy.constants import LEN_CONV

from edftpy.engine.engine import Engine

try:
    __version__ = pyec.__version__
except Exception :
    __version__ = '0.0.1'

class EngineEnviron(Engine):
    """Engine for Environ

    For now, just inherit the Engine, since the Environ driver
    is reliant on the qepy module
    """
    def __init__(self, **kwargs):
        """
        this mirrors QE unit conversion, Environ has atomic internal units
        """
        unit_len = kwargs.get('length', 1.0)
        unit_vol = unit_len ** 3
        units = kwargs.get('units', {})
        kwargs['units'] = units
        kwargs['units']['volume'] = unit_vol
        kwargs['units']['energy'] = 0.5
        super().__init__(**kwargs)

        # initialize some persistent objects that Environ needs
        # TODO this should be inferred through edftpy
        self.nat = None
        # this being persistent takes up memory, might want a better way of
        # temporariliy storing and then cleaning this up
        self.potential = None

    def get_forces(self, **kwargs):
        """get Environ force contribution
        """
        force = np.zeros((3, self.nat), dtype=float, order='F')
        environ_calc.calc_force(force)
        return force.T

    def calc_energy(self, **kwargs):
        """get Environ energy contribution
        """
        # move these array options to pyec maybe
        energy = environ_calc.calc_energy()
        return energy

    def initial(self, inputfile= None, comm = None, **kwargs):
        """initialize the Environ module

        Args:
            comm (uint, optional): communicator for MPI. Defaults to None.
        """
        # TODO verify units for values that get communicated to Environ from edftpy
        # TODO handle any input value missing things better?
        #-----------------------------------------------------------------------
        kwargs = self.write_input(**kwargs)
        #-----------------------------------------------------------------------
        self.nat = kwargs.get('nat')
        ntyp = kwargs.get('ntyp')
        nelec = kwargs.get('nelec')
        atom_label = kwargs.get('atom_label')
        alat = kwargs.get('alat')
        at = kwargs.get('at')
        gcutm = kwargs.get('gcutm')
        # if gcutm not supplied, we can infer its value with ecutrho
        if gcutm is None:
            ecutrho = kwargs.get('ecutrho')
            alat = kwargs.get('alat')
            _check_kwargs('ecutrho', ecutrho, 'initial')
            gcutm = ecutrho / (2 * np.pi / alat) ** 2
        ityp = kwargs.get('ityp')
        zv = kwargs.get('zv')
        tau = kwargs.get('tau')
        do_comp_mt = kwargs.get('do_comp_mt')
        # rho = kwargs.get('rho') # maybe make this a named argument to parallel the scf function

        # raise Exceptions here if the keywords are not supplied
        _check_kwargs('nat', self.nat, 'initial')
        _check_kwargs('ntyp', ntyp, 'initial')
        _check_kwargs('nelec', nelec, 'initial')
        _check_kwargs('atom_label', atom_label, 'initial')
        _check_kwargs('at', at, 'initial')
        _check_kwargs('ityp', ityp, 'initial')
        _check_kwargs('zv', zv, 'initial')
        _check_kwargs('tau', tau, 'initial')
        _check_kwargs('do_comp_mt', do_comp_mt, 'initial')

        self.comm = comm or self.comm

        if hasattr(comm, 'py2f') :
            commf = comm.py2f()
        else :
            commf = None
        # TODO program unit needs to be set externally perhaps
        iounit = 6
        environ_setup.init_io(comm.rank == 0, 0, commf, iounit)
        environ_setup.read_input()
        environ_setup.init_environ(
                commf, nelec, self.nat, ntyp, atom_label[:, :ntyp], ityp, zv,
                do_comp_mt, alat, at, gcutm)
        environ_control.update_ions(self.nat, tau, alat)
        environ_control.update_cell(at, alat)
        nnr = environ_calc.get_nnt()

        if comm is None or comm.rank == 0 :
            self.potential = np.zeros((nnr,), dtype=float)
        else :
            self.potential = np.zeros((1,), dtype=float)

        # TODO reconsider these steps
        # if rho is not None:
        #     environ_control.update_electrons(rho, lscatter=True)
        #     environ_calc.calc_potential(False, self.potential, lgather=True) # might not be necessary?
        # else:
        #     # print("electrons not initialized until scf")
        #     rho = np.zeros((environ_calc.get_nnt(), 1,), dtype=float, order='F')

    def write_input(self, subcell = None, **kwargs):
        defaults = {
                'ecutrho' : 300,
                }
        nat = subcell.ions.nat
        ntyp = len(subcell.ions.zval)
        alat = subcell.grid.latparas[0]
        at = subcell.ions.cell/alat
        tau = subcell.ions.positions().T / subcell.grid.latparas[0]
        labels = subcell.ions.get_chemical_symbols()
        zv = np.zeros(ntyp)
        # atom_label = np.ones((3, ntyp), dtype = 'int32')*32
        atom_label = np.zeros((3, ntyp), dtype = 'c')
        atom_label[:] = ' '
        ityp = np.ones(nat, dtype = 'int32')
        i = -1
        nelec = 0.0
        for key, v in subcell.ions.zval.items():
            i += 1
            zv[i] = v
            atom_label[:len(key), i] = key
            mask = labels == key
            ityp[mask] = i + 1
            nelec += v*np.count_nonzero(mask)
        subs = {
                'at' : at,
                'nat' : nat,
                'ntyp' : ntyp,
                'alat' : alat,
                'tau' : tau,
                'zv' : zv,
                'atom_label' : atom_label,
                'ityp' : ityp,
                'nelec' : nelec,
                }
        defaults.update(kwargs)
        defaults.update(subs)
        return defaults

    def scf(self, rho, update = True, **kwargs):
        """A single electronic step for Environ

        Args:
            rho (np.ndarray): the density object
        """
        environ_control.update_electrons(rho, lscatter=True)
        if self.potential is None:
            raise ValueError("`potential` not initialized, has `initial` been run?")
        environ_calc.calc_potential(update, self.potential, lgather=True)

    def get_potential(self, **kwargs):
        """Returns the potential from Environ
        """
        return self.potential

    def set_mbx_charges(self, rho):
        """Supply Environ with MBX charges
        """
        environ_control.add_mbx_charges(rho, lscatter=True)

    def get_grid(self, nr, **kwargs):
        nr[0] = environ_calc.get_nr1x()
        nr[1] = environ_calc.get_nr2x()
        nr[2] = environ_calc.get_nr3x()
        return nr

    def update_ions(self, subcell = None, update = 0, **kwargs):
        environ_setup.clean_environ()
        self.write_input(subcell = subcell, **kwargs)
        self.initial(comm = self.comm)

    def end_scf(self, **kwargs):
        pass

    def stop_run(self, **kwargs):
        environ_setup.clean_environ()

def _check_kwargs(key, val, funlabel='\b'):
    """check that a kwargs value is set

    Args:
        key (str): the key for printing the label
        val (Any): only check for None
        funlabel (str, optional): function name. Defaults to '\b'.

    Raises:
        ValueError: [description]
    """
    if val is None:
        raise ValueError(f'`{key}` not set in {funlabel} function')
