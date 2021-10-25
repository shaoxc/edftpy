import pyec
import pyec.setup_interface as setup
import pyec.control_interface as controller
import pyec.calc_interface as calculator
import pyec.output_interface as output

import numpy as np

from dftpy.constants import LEN_CONV

from edftpy.engine.engine import Engine
from edftpy.io import print2file

try:
    __version__ = pyec.__version__
except Exception:
    __version__ = '0.0.1'


class EngineEnviron(Engine):
    """Engine for Environ."""
    def __init__(self, **kwargs) -> None:

        self.nat = 0
        self.threshold = 0.0
        self.potential = None  # TODO does this need to be persist?
        self.outfile = 'environ.out'

        if kwargs.get('append', False):
            self.fileobj = open(self.outfile, 'a')
        else:
            self.fileobj = open(self.outfile, 'w')

    # TODO move initial into __init__

    @print2file()
    def initial(self,
                inputfile: str = 'environ.in',
                comm=None,
                **kwargs) -> None:
        """
        Initialize the Environ module.

        :param inputfile: Environ input filename
        :type  inputfile: str, optional
        :param comm     : communicator for MPI, defaults to None
        :type  comm     : uint, optional

        :raises ValueError: if missing necessary input parameters in kwargs
        """
        # EXPECTED INPUT
        inputs = dict.fromkeys([
            'nelec', 'nat', 'ntyp', 'atom_label', 'ityp', 'zv',
            'use_internal_pbc_corr', 'alat', 'at', 'tau'
        ])

        # INPUT VALIDATION
        for key in inputs.keys():
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f'`{key}` not set in initial function')
            else:
                inputs[key] = value

        # PERSISTENT PARAMETERS
        self.nat = inputs['nat']
        self.alat = inputs['alat']
        self.comm = comm

        # SET G-VECTOR CUTOFF
        gcutm = kwargs.get('gcutm')
        if not gcutm:
            ecutrho = kwargs.get('ecutrho')
            if value is None:
                raise ValueError(f'`{key}` not set in initial function')
            gcutm = ecutrho / (2 * np.pi / self.alat)**2

        # COMMUNICATOR
        ionode = comm.rank == 0
        if hasattr(comm, 'py2f'):
            commf = comm.py2f()
        else:
            commf = None

        # DISTRIBUTE INPUT
        ions_input = {key: inputs.get(key) for key in ('nat', 'tau', 'alat')}
        cell_input = {key: inputs.get(key) for key in ('at', 'alat')}
        inputs.update({'comm': commf, 'gcutm': gcutm})
        del inputs['tau']

        # INITIALIZE ENVIRON
        setup.init_io(ionode, 0, commf, 6)
        setup.read_input(inputfile)
        setup.init_environ(**inputs)

        # # UPDATE IONS AND CELL
        controller.update_ions(**ions_input)
        controller.update_cell(**cell_input)

        self.threshold = controller.get_threshold()

        self.nnt = controller.get_nnt()  # total number of FFT grid points
        self.potential = np.zeros(self.nnt, order='F')

    @print2file()
    def update_ions(self, subcell=None, update: int = 0, **kwargs) -> None:
        setup.clean_environ()
        self.write_input(subcell=subcell, **kwargs)
        self.initial(comm=self.comm)

    def pre_scf(self, rho) -> None:
        """
        Updates electrons and potential before starting SCF cycle.

        :param rho   : the density object
        :type  rho   : ndarray[float]
        """
        restart = controller.is_restart()
        self.scf(rho, restart)

    @print2file()
    def scf(self, rho, update: bool = True, **kwargs) -> None:
        """
        Runs a single electronic step for Environ.

        :param rho   : the density object
        :type  rho   : ndarray[float]
        :param update: if Environ should compute contribution to potential
        :type  update: bool

        :raises ValueError: if potential is not initialized
        """
        if self.potential is None:
            raise ValueError("Potential not initialized.")

        controller.update_electrons(rho, True)
        calculator.calc_potential(update, self.potential, lgather=True)

    @print2file()
    def get_force(self, **kwargs):
        """
        Returns Environ's contribution to the force.

        :return: Environ's contribution to the force
        :rtype : ndarray[float]

        :raises ValueError: if missing number of atoms (not initialized)
        """
        if not self.nat: raise ValueError("Environ not yet initialized.")

        force = np.zeros((3, self.nat), dtype=float, order='F')
        calculator.calc_force(force)

        return force.T  # * self.units['energy']

    def get_potential(self, **kwargs):
        """
        Returns Environ's contribution to the potential.

        :return: Environ's contribution to the potential
        :rtype : ndarray[float]

        :raises ValueError: if missing potential (not initialized)
        """
        return self.potential  # * self.units['energy']

    def get_nnt(self) -> int:
        """
        Returns total number of grid points in Environ.

        :return: total number of grid points in Environ
        :rtype : int
        """
        return self.nnt

    def get_grid(self, nr, **kwargs):
        """
        Returns Environ's grid dimensions.

        :return: the dimensions of Environ's FFT grid
        :rtype : ndarray[int]
        """
        nr[0] = calculator.get_nr1x()
        nr[1] = calculator.get_nr2x()
        nr[2] = calculator.get_nr3x()
        return nr

    def get_threshold(self) -> float:
        """
        Returns Environ's scf threshold.

        :return: Environ's scf threshold
        :rtype : float
        """
        return self.threshold

    @print2file()
    def set_mbx_charges(self, rho) -> None:
        """
        Add MBX charges to Environ's charge density.
        
        :param rho: the MBX charge density
        :type  rho: ndarray[float]
        """
        controller.add_mbx_charges(rho, True)

    @print2file()
    def calc_energy(self, **kwargs) -> float:
        """
        Compute Environ's contribution to the energy.

        :return: Environ's contribution to the energy
        :rtype : float
        """
        energy = calculator.calc_energy()
        return energy  # * self.units['energy']

    def print_energies(self) -> None:
        """Prints Environ's contribution to the energy."""
        output.print_energies()

    def print_potential_shift(self) -> None:
        """Prints potential shift due to the use of Gaussian-spread ions."""
        output.print_potential_shift()

    def end_scf(self, **kwargs) -> None:
        pass

    @print2file()
    def stop_run(self, **kwargs) -> None:
        """Destroy Environ objects."""
        setup.clean_environ()
