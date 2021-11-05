import mbx

import numpy as np
from dftpy.constants import LEN_CONV

from edftpy.engine.engine import Engine
from edftpy.io import print2file

try:
    __version__ = mbx.__version__
except Exception :
    __version__ = '0.0.1'

MBX_MONOMERS = {
        "li"   : ["Li"],
        "na"   : ["Na"],
        "k"    : ["K"],
        "rb"   : ["Rb"],
        "cs"   : ["Cs"],
        "f"    : ["F"],
        "cl"   : ["Cl"],
        "br"   : ["Br"],
        "i"    : ["I"],
        "ar"   : ["Ar"],
        "he"   : ["He"],
        "h2"   : ["H", "H"],
        "h2o"  : ["O", "H", "H"],
        "co2"  : ["C", "O", "O"],
        "nh3"  : ["N", "H", "H", "H"],
        "nh4+" : ["N", "H", "H", "H", "H"],
        "ch4"  : ["C", "H", "H", "H", "H"],
        "pf6-" : ["P", "F", "F", "F", "F", "F", "F"],
        "n2o5" : ["O", "N", "N", "O", "O", "O", "O"],
        }

class EngineMBX(Engine):
    """Engine for MBX
    units :
        Length : Angstrom
        Energy : ??
    """
    def __init__(self, **kwargs):
        units = kwargs.get('units', {})
        unit_len = LEN_CONV["Angstrom"]["Bohr"]
        # units['length'] = units.get('length', mbx.MBXLENGTH2AU)
        # units['energy'] = units.get('energy', mbx.MBXENERGY2AU)
        units['length'] = units.get('length', unit_len)
        units['energy'] = units.get('energy', 1.0)
        units['order'] = 'C'
        kwargs['units'] = units
        super().__init__(**kwargs)

        self.inputs = {}

    def get_force(self, **kwargs):
        force = None
        return force.T * self.units['energy']/self.units['length']

    def get_energy(self, olevel = 0, **kwargs) -> float:
        energy = mbx.get_energy_pbc_nograd(self.positions, len(self.labels), self.box)
        return energy * self.units['energy']

    @print2file()
    def initial(self, inputfile = 'mbx.json', comm=None, **kwargs):
        inputs = self.write_input(**kwargs)
        self.positions = inputs['positions']
        self.monomer_natoms = inputs['monomer_natoms']
        self.labels = inputs['labels']
        self.monomer_names = inputs['monomer_names']
        self.inputfile = inputfile
        self.box = inputs['box']
        mbx.initialize_system(self.positions, self.monomer_natoms, self.labels, self.monomer_names, self.inputfile)
        # mbx.initialize_system(xyz,number_of_atoms_per_monomer,atom_names,monomer_names,json_file)
        #-----------------------------------------------------------------------
        # self.npoints = len(self.labels) + len(self.monomer_names) # ?? #atoms+#monomers ??
        self.npoints = len(self.monomer_names)*4 # ?? #monomers ??
        # #-----------------------------------------------------------------------
        self.points_mm = mbx.get_xyz(self.npoints)
        self.points_mm = np.array(self.points_mm).reshape((-1, 3))
        # ?? crystal coordinates or just scaled
        self.points_mm[:, 0] /= self.box[0]
        self.points_mm[:, 1] /= self.box[3]
        self.points_mm[:, 2] /= self.box[6]

    def initial_old(self, inputfile = 'mbx.json', comm = None, subcell = None, grid = None, **kwargs):
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
        json_file = inputfile
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

    @print2file()
    def update_ions(self, subcell=None, **kwargs) :
        # Can we use this update ions?
        mbx.set_coordinates(self.positions, len(self.labels))
        mbx.set_box(self.box)

    def scf(self, **kwargs):
        pass

    def get_potential(self, out = None, grid = None, **kwargs):
        grid_points = grid.r.reshape((3, -1)).T
        grid_points = grid_points.ravel()
        pot, potfield = mbx.get_potential_and_electric_field_on_points(grid_points, grid_points.size//3)
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

    def write_input(self, subcell = None, monomers = {}, **kwargs):
        """
        Note :
            The order of atoms should be same as atoms in the monomers
        TODO :
            add a list contains the indices of the atoms in the monomers
        """
        if subcell is None : return kwargs

        if len(monomers) < 1 :
            monomers = MBX_MONOMERS
        monomers = sorted(monomers.items(), key=lambda d : len(d[1]), reverse=True)

        labels = subcell.ions.labels.tolist()
        natoms = len(labels)

        monomer_names = []
        monomer_natoms = []
        index = 0
        for i in range(natoms):
            for key, mon in monomers :
                nat = len(mon)
                symbols = labels[index:index + nat]
                if mon == symbols :
                    monomer_names.append(key)
                    monomer_natoms.append(nat)
                    index += nat
                    break
            if index==natoms : break
        if index < natoms: raise AttributeError("Please check the monomers and input atoms")

        inputs = kwargs
        inputs['labels'] = labels
        inputs['positions'] = subcell.ions.pos.to_cart().ravel()/self.units['length']
        inputs['monomer_names'] = monomer_names
        inputs['monomer_natoms'] = monomer_natoms
        if inputs.get('box', None) is None :
            inputs['box'] = subcell.ions.pos.cell.lattice.ravel()/self.units['length']
        return inputs
