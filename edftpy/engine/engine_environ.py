import sys

import pyec.setup_interface as environ_setup
import pyec.control_interface as environ_control
import pyec.calc_interface as environ_calc
import pyec.output_interface as environ_output

import numpy as np

from dftpy.constants import LEN_CONV

from edftpy.engine.driver import Engine

class EngineEnviron(Engine):
    """Engine for Environ
    
    For now, just inherit the Engine, since the Environ driver
    is reliant on the qepy module
    """
    def __init__(self, **kwargs):
        """
        this mirrors QE unit conversion, Environ has atomic internal units
        """
        bohr_radius_si = kwargs.get('bohr_radius_si', 1.0)
        unit_len = LEN_CONV["Bohr"]["Angstrom"] / bohr_radius_si / 1E10
        unit_vol = unit_len ** 3
        units = kwargs.get('units', {})
        kwargs['units'] = units
        kwargs['units']['volume'] = unit_vol
        super().__init__(**kwargs)

        # initialize some persistent objects that Environ needs
        # TODO this should be inferred through edftpy
        self.nat = None
        # this being persistent takes up memory, might want a better way of
        # temporariliy storing and then cleaning this up
        self.dvtot = None
        
    def get_force(self, **kwargs):
        """get Environ force contribution
        """
        force = np.zeros((3, self.nat), dtype=float, order='F')
        environ_calc.calc_force(force)
        return force.T * 0.5 # Ry to a.u.

    def calc_energy(self, **kwargs):
        """get Environ energy contribution
        """
        # move these array options to pyec maybe
        energy = environ_calc.calc_energy()
        return energy * 0.5 # Ry to a.u.

    def initial(self, comm = None, **kwargs):
        """initialize the Environ module

        Args:
            comm (uint, optional): communicator for MPI. Defaults to None.
        """
        # TODO verify units for values that get communicated to Environ from edftpy
        # TODO handle any input value missing things better?
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
        e2 = kwargs.get('e2')
        ityp = kwargs.get('ityp')
        zv = kwargs.get('zv')
        tau = kwargs.get('tau')
        vltot = kwargs.get('vltot')
        rho = kwargs.get('rho') # maybe make this a named argument to parallel the scf function

        # raise Exceptions here if the keywords are not supplied
        _check_kwargs('nat', self.nat, 'initial')
        _check_kwargs('ntyp', ntyp, 'initial')
        _check_kwargs('nelec', nelec, 'initial')
        _check_kwargs('atom_label', atom_label, 'initial')
        _check_kwargs('at', at, 'initial')
        _check_kwargs('e2', e2, 'initial')
        _check_kwargs('ityp', ityp, 'initial')
        _check_kwargs('zv', zv, 'initial')
        _check_kwargs('tau', tau, 'initial')

        # TODO program unit needs to be set externally perhaps
        environ_setup.init_io('PW', True, 0, comm, 6) 
        environ_setup.init_base_first(nelec, self.nat, ntyp, atom_label[:, :ntyp], False)
        environ_setup.init_base_second(alat, at, comm, 0, 0, gcutm, e2)
        environ_control.update_ions(self.nat, ntyp, ityp, zv[:ntyp], tau, alat)
        environ_control.update_cell(at, alat)

        self.dvtot = np.zeros((environ_calc.get_nnt(),), dtype=float)

        if vltot is not None:
            environ_control.update_potential(vltot)
        else:
            print("potential not initialized until scf")
        # TODO reconsider these steps
        if rho is not None: 
            environ_control.update_electrons(rho, lscatter=True)
            environ_calc.calc_potential(False, self.dvtot, lgather=True) # might not be necessary?
        #else:
        #    print("electrons not initialized until scf")
        #    rho = np.zeros((environ_calc.get_nnt(), 1,), dtype=float, order='F')

    def scf(self, rho, **kwargs):
        """A single electronic step for Environ

        Args:
            rho (np.ndarray): the density object
        """
        environ_control.update_electrons(rho, lscatter=True)
        if self.dvtot is None:
            raise ValueError(f'`dvtot` not initialized, has `initial` been run?')
        environ_calc.calc_potential(True, self.dvtot, lgather=True)

    def get_potential(self):
        """Returns the potential from Environ
        """
        return self.dvtot

    def clean(self):
        """Clean up memory on the Fortran side
        """
        environ_setup.environ_clean(True)

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
