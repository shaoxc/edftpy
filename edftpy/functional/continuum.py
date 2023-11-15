from edftpy.utils.common import Functional, AbsFunctional
from ase.atoms import symbols2numbers


class Environ(AbsFunctional):
    def __init__(self, **kwargs):
        self.initial(**kwargs)

    def initial(self, file = 'environ.yml', grid=None, ions=None, **kwargs) -> None:
        """Initialize the Environ module."""

        from envyron.domains import EnvironGrid
        # from envyron.representations import EnvironDensity
        from envyron.io.input import Input
        from envyron.setup import Setup
        from envyron.main import Main
        from envyron.calculator import Calculator
        #
        self.ions = ions
        self.grid = grid
        #
        zval = ions.zval
        natoms = len(ions.numbers)
        ntypes = len(zval)
        ion_labels = ions.symbols_uniq
        ion_ids = symbols2numbers(ion_labels)
        itypes = [ion_ids.index(id) for id in ions.numbers]
        zv = [zval[k] for k in ion_labels]

        # DFTpy Grid to EnvironGrid
        grid = EnvironGrid(grid.cell, nr=grid.nrR, label='system')

        my_input = Input(natoms=natoms, filename=file)
        my_setup = Setup(my_input)
        my_setup.init_cell(grid)
        my_setup.init_numerical(False)

        environ = Main(my_setup,natoms,ntypes,itypes,zv,ion_ids)
        environ.update_cell_dependent_quantities()
        environ.update_ions(ions.positions)

        self.driver = Calculator(environ)

    def update_ions(self, ions=None, update = 0, **kwargs) -> None:
        """Update ion positions."""
        self.ions = ions
        self.driver.main.update_ions(ions.positions)

    def update_electrons(self, density, update = True, **kwargs) -> None:
        """Run a single electronic step for Environ."""
        self.driver.main.update_electrons(density)

    def get_potential(self, **kwargs) :
        """Return Environ's contribution to the potential."""
        self.driver.potential(True)
        return self.driver.main.vsoftcavity * 0.5

    def get_energy(self, **kwargs) -> float:
        """Compute Environ's contribution to the energy."""
        self.driver.energy()
        return self.driver.main.evolume * 0.5

    def stop_run(self, **kwargs) -> None:
        """Destroy Environ objects."""
        raise AttributeError("Should implemented in Environ.")

    def __call__(self, density, calcType=["E","V"], **kwargs):
        return self.compute(density, calcType=calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], **kwargs):
        obj = Functional(name = 'Environ')
        self.update_electrons(density)
        if 'V' in calcType : obj.potential = self.get_potential(**kwargs)
        if 'E' in calcType : obj.energy = self.get_energy(**kwargs)
        return obj

    def forces(self, density, pseudo = None, **kwargs):
        raise AttributeError("Should implemented in Environ.")

    def stress(self, density, **kwargs):
        raise AttributeError("Should implemented in Environ.")
