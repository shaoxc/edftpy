import mbx

import numpy as np
# from dftpy.constants import LEN_CONV
from edftpy.engine.engine import Engine

try:
    __version__ = mbx.__version__
except Exception :
    __version__ = '0.0.1'

class EngineMBX(Engine):
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
        super().__init__(**kwargs)

    def get_force(self, **kwargs):
        force = None
        return force

    def calc_energy(self, *args, **kwargs):
        energy = 0.0
        return energy

    def initial(self, filename = None, comm = None, subcell = None, grid = None, **kwargs):
        """
        Note :
            The atoms should always be 'O H H O H H...'
        """
        #-----------------------------------------------------------------------
        natom = subcell.ions.nat
        nmol = natom//3
        number_of_atoms_per_monomer = [3,]*nmol
        atom_names = ["O","H","H"]* nmol
        monomer_names = ["h2o",] * nmol
        # json_file = "./mbx.json"
        json_file = filename
        xyz = subcell.ions.pos.ravel()
        box = subcell.ions.pos.cell.lattice.ravel()
        #-----------------------------------------------------------------------
        mbx.initialize_system(xyz,number_of_atoms_per_monomer,atom_names,monomer_names,json_file)
        mbx.set_coordinates(xyz, natom)
        mbx.set_box(box)
        #-----------------------------------------------------------------------
        self.subcell = subcell
        self.npoints = nmol*4
        #-----------------------------------------------------------------------
        self.points_mm = mbx.get_xyz(self.npoints)
        self.points_mm = np.array(self.points_mm).reshape((-1, 3))/subcell.grid.latparas

    def scf(self, **kwargs):
        pass

    def get_potential(self, out = None, grid = None, **kwargs):
        grid_points = grid.r.reshape((3, -1)).T
        grid_points = grid_points.ravel()
        pot, potfield = mbx.get_potential_and_electric_field_on_points(grid_points, grid_points.size//3)
        pot = np.asarray(pot)
        return pot.reshape(self.nr)

    def set_extpot(self, extpot = None, **kwargs):
        # pot = extpot.get_value_at_points(self.points_mm).ravel()
        pot = self.get_value_at_points(extpot, self.points_mm).ravel()
        extfield = extpot.gradient()
        potfield = []
        for i in range(3):
            p = self.get_value_at_points(extfield[i], self.points_mm)
            potfield.append(p)
        potfield= np.asarray(potfield).ravel()
        mbx.set_potential_and_electric_field_on_sites(pot, potfield)

    def get_value_at_points(self, data, points):
        values = np.empty(points.shape[0])
        for i, p in enumerate(points):
            ip = (np.rint(p*data.grid.nrR)).astype('int32')
            values[i] = data[ip[0], ip[1], ip[2]]
        return values

    def get_grid(self, nr, gaps = 4.5, **kwargs):
        nr[:] = self.subcell.grid.latparas / gaps
        self.nr = nr
        return nr

    def get_charge(self, charge, **kwargs):
        raise AttributeError("Will do")
        return charge
