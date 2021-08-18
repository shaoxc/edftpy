import pyec.pyec_interface as environ

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
        # TODO this should be rethought, maybe have a container
        self.nnr = None
        self.nelec = None
        # this being persistent takes up memory, might want a better way of
        # temporariliy storing and then cleaning this up
        self.dvtot = None
        
    def get_force(self, **kwargs):
        """get Environ force contribution
        """
        pass # TODO test and implement forces in Environ

    def get_energy(self, **kwargs):
        """get Environ energy contribution
        """
        # move these array options to pyec maybe
        energy = np.zeros((1,), dtype=float)
        environ.calc_energy(energy)
        return energy[0]

    def initial(self, comm = None, **kwargs):
        """initialize the Environ module

        Args:
            comm (int, optional): communicator for MPI. Defaults to None.
        """
        # TODO verify units for values that get communicated to Environ from edftpy
        # TODO handle any input value missing things better?
        nat = kwargs.get('nat')
        ntyp = kwargs.get('ntyp')
        self.nelec = kwargs.get('nelec')
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
        self.nnr = kwargs.get('nnr')
        vltot = kwargs.get('vltot')
        rho = np.zeros((nnr, 1,), dtype=float, order='F')
        self.dvtot = np.zeros((nnr,), dtype=float)

        # raise Exceptions here if the keywords are not supplied
        _check_kwargs('nat', nat, 'initial')
        _check_kwargs('ntyp', ntyp, 'initial')
        _check_kwargs('nelec', self.nelec, 'initial')
        _check_kwargs('atom_label', atom_label, 'initial')
        _check_kwargs('at', at, 'initial')
        _check_kwargs('e2', e2, 'initial')
        _check_kwargs('ityp', ityp, 'initial')
        _check_kwargs('zv', zv, 'initial')
        _check_kwargs('tau', tau, 'initial')
        _check_kwargs('nnr', self.nnr, 'initial')
        _check_kwargs('vltot', vltot, 'initial')

        # TODO program unit needs to be set externally perhaps
        environ.init_io('PW', True, 0, comm, 6) 
        environ.init_base_first(self.nelec, nat, ntyp, atom_label[:, :ntyp], False)
        environ.init_base_second(alat, at, comm, 0, 0, gcutm, e2)
        environ.init_ions(nat, ntyp, ityp, zv, tau, alat)
        environ.init_cell(at, alat)
        environ.init_potential(self.nnr, vltot)
        # TODO reconsider these steps
        environ.init_electrons(self.nnr, rho, self.nelec)
        environ.calc_potential(False, self.nnr, self.dvtot) # might not be necessary?

    def scf(self, rho, **kwargs):
        """A single electronic step for Environ

        Args:
            rho (np.ndarray): the density object
        """
        environ.init_electrons(self.nnr, rho, self.nelec)
        if self.dvtot is None:
            raise ValueError(f'`dvtot` not initialized, has `initial` been run?')
        environ.calc_potential(True, self.nnr, self.dvtot)

    def get_potential(self):
        """Returns the potential from Environ
        """
        return self.dvtot

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
