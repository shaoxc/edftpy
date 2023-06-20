from typing import Any, Dict
import pyec
import pyec.environ_interface as environ

import numpy as np

from edftpy.engine.engine import Engine
from edftpy.io import print2file

try:
    __version__ = pyec.__version__
except Exception:
    __version__ = '0.0.1'


class EngineEnviron(Engine):
    """Engine for Environ."""
    def __init__(self, **kwargs) -> None:
        units = kwargs.get('units', {})
        kwargs['units'] = units
        unit_len = units.get('length', 1.0)
        unit_ene = units.get('energy', 0.5)
        unit_vol = unit_len**3
        kwargs['units']['volume'] = unit_vol
        kwargs['units']['energy'] = unit_ene
        kwargs['units']['order'] = 'F'
        super().__init__(**kwargs)

        self.nat = 0
        self.potential = None

    @print2file()
    def initial(self,
                inputfile: str = 'environ.in',
                comm=None,
                **kwargs) -> None:
        """Initialize the Environ module."""

        # EXPECTED INPUT
        kwargs = self.get_input(**kwargs)

        inputs = dict.fromkeys([
            'nelec', 'nat', 'ntyp', 'atom_label', 'ityp', 'zv',
            'use_internal_pbc_corr', 'at', 'tau'
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
        self.comm = comm

        # SET G-VECTOR CUTOFF
        gcutm = kwargs.get('gcutm')
        if not gcutm:
            ecutrho = kwargs.get('ecutrho')
            if value is None:
                raise ValueError(f'`{key}` not set in initial function')
            gcutm = ecutrho / (2 * np.pi)**2

        # COMMUNICATOR
        if hasattr(comm, 'py2f'):
            commf = comm.py2f()
            is_ionode = comm.rank == 0
        else:
            commf = None
            is_ionode = True

        # UPDATE INPUT
        inputs.update({'comm': commf, 'gcutm': gcutm})
        tau = inputs['tau']
        del inputs['tau']

        # INITIALIZE ENVIRON
        environ.init_io(is_ionode, 0, commf, 6)
        environ.read_input(inputfile)
        environ.init_environ(**inputs)

        # UPDATE IONS AND CELL
        environ.update_ions(self.nat, tau)
        environ.update_cell(inputs['at'])

        self.nnt = environ.get_nnt()  # total number of FFT grid points
        self.potential = np.zeros(self.nnt, order='F')

    @print2file()
    def update_ions(self, subcell=None, update: int = 0, **kwargs) -> None:
        """Update ion positions."""
        if not subcell:
            raise ValueError("Missing subcell.")

        tau = subcell.ions.pos.to_cart().T
        environ.update_ions(self.nat, tau)

    @print2file()
    def scf(self, rho: np.ndarray, update: bool = True, **kwargs) -> None:
        """Run a single electronic step for Environ."""
        if self.potential is None:
            raise ValueError("Potential not initialized.")

        environ.update_electrons(rho, True)
        environ.calc_potential(update, self.potential, lgather=True)

    @print2file()
    def get_forces(self, **kwargs) -> np.ndarray:
        """Return Environ's contribution to the force."""
        if not self.nat:
            raise ValueError("Environ not yet initialized.")

        force = np.zeros((3, self.nat), dtype=float, order='F')
        environ.calc_force(force)

        return force.T * self.units['energy']

    def get_potential(self, **kwargs) -> np.ndarray:
        """Return Environ's contribution to the potential."""
        return self.potential * self.units['energy']

    def get_nnt(self) -> int:
        """Return total number of grid points in Environ."""
        return self.nnt

    def get_grid(self, nr: np.ndarray = None, **kwargs) -> np.ndarray:
        """Return Environ's grid dimensions."""
        if nr is None:
            nr = np.zeros(3, dtype='int32')

        for i in range(3):
            nr[i] = environ.get_nrx(i)

        return nr

    @print2file()
    def set_mbx_charges(self, rho: np.ndarray) -> None:
        """Add MBX charges to Environ's charge density."""
        environ.add_mbx_charges(rho, True)

    def calc_energy(self, **kwargs):
        return self.get_energy(self, **kwargs)

    @print2file()
    def get_energy(self, olevel=0, **kwargs) -> float:
        """Compute Environ's contribution to the energy."""
        if olevel == 0:
            energy = environ.calc_energy()
        else:
            energy = 0.0

        return energy * self.units['energy']

    def print_energies(self) -> None:
        """Print Environ's contribution to the energy."""
        environ.print_energies()

    def print_potential_shift(self) -> None:
        """Print potential shift due to the use of Gaussian-spread ions."""
        environ.print_potential_shift()

    @print2file()
    def stop_run(self, **kwargs) -> None:
        """Destroy Environ objects."""
        environ.clean_environ()

    def get_input(self, subcell=None, **kwargs) -> Dict[str, Any]:
        """Build Environ input."""
        defaults = {
            'ecutrho': 300,
            'use_internal_pbc_corr': False,
        }
        defaults.update(kwargs)

        if subcell is None: return defaults  # if running QEpy+PyE-C directly

        # Update base on the subcell
        nat = subcell.ions.nat
        ntyp = len(subcell.ions.zval)
        alat = subcell.grid.latparas[0]
        at = subcell.ions.cell/alat
        tau = subcell.ions.positions().T / subcell.grid.latparas[0]
        labels = subcell.ions.get_chemical_symbols()
        zv = np.zeros(ntyp)
        atom_label = np.zeros((3, ntyp), dtype='c')
        atom_label[:] = ' '
        ityp = np.ones(nat, dtype='int32')

        i = -1
        nelec = 0.0
        # COLLECT ATOM LABELS, IONIC CHARGES, AND NUMBER OF ELECTRONS
        for key, v in subcell.ions.zval.items():
            i += 1
            zv[i] = v
            atom_label[:len(key), i] = key
            mask = labels == key
            ityp[mask] = i + 1
            nelec += v * np.count_nonzero(mask)

        subs = {
            'at': at,
            'nat': nat,
            'ntyp': ntyp,
            'tau': tau,
            'zv': zv,
            'atom_label': atom_label,
            'ityp': ityp,
            'nelec': nelec,
        }

        defaults.update(subs)

        return defaults

    @staticmethod
    def get_kwargs_from_qepy(qepy=None) -> Dict[str, Any]:
        """
        For direct QEpy+PyE-C, return QEpy parameters necessary in Environ.
        """
        if qepy is None: import qepy

        nat = qepy.ions_base.get_nat()  # number of atoms
        ntyp = qepy.ions_base.get_nsp()  # number of species
        nelec = qepy.klist.get_nelec()  # number of electrons
        atom_label = qepy.ions_base.get_array_atm().view('c')  # atom labels
        atom_label = atom_label[:, :ntyp]  # discard placeholders
        at = qepy.cell_base.get_array_at()  # 3 x 3 lattice in atomic units
        gcutm = qepy.gvect.get_gcutm()  # G-vector cutoff
        ityp = qepy.ions_base.get_array_ityp()  # species indices
        zv = qepy.ions_base.get_array_zv()  # ionic charges
        tau = qepy.ions_base.get_array_tau()  # ion positions

        # CONCERT TO ATOMIC UNITS
        alat = qepy.cell_base.get_alat()  # lattice parameter
        at = at * alat
        tau = tau * alat
        gcutm = gcutm / alat**2

        kwargs = {
            'nat': nat,
            'ntyp': ntyp,
            'nelec': nelec,
            'atom_label': atom_label,
            'ityp': ityp,
            'at': at,
            'gcutm': gcutm,
            'zv': zv,
            'tau': tau,
            'use_internal_pbc_corr': False,
        }

        return kwargs
