import mbx

import numpy as np
from dftpy.constants import LEN_CONV

from edftpy.engine.engine import Engine
from edftpy.io import print2file
from edftpy.mpi import sprint

try:
    __version__ = mbx.__version__
except Exception :
    __version__ = '0.0.1'

MBX_MONOMERS = {
        "li+"   : ["Li"],
        "na+"   : ["Na"],
        "k+"    : ["K"],
        "rb+"   : ["Rb"],
        "cs+"   : ["Cs"],
        "f-"    : ["F"],
        "cl-"   : ["Cl"],
        "br-"   : ["Br"],
        "i-"    : ["I"],
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
        Energy : kcal/mol

    Note :
        Only works for serial.
    """
    def __init__(self, xc = 'mbx', **kwargs):
        units = kwargs.get('units', {})
        # units['length'] = units.get('length', LEN_CONV["Angstrom"]["Bohr"])
        # # units['length'] = units.get('length', mbx.MBXLENGTH2AU)
        # units['energy'] = units.get('energy', mbx.MBXENERGY2AU)
        units['length'] = units.get('length', 1.0)
        units['energy'] = units.get('energy', 1.0)
        units['order'] = 'C'
        kwargs['units'] = units
        super().__init__(**kwargs)
        self.xc = xc
        if isinstance(self.xc, dict):
            self.xc = self.xc.get('xc', 'mbx') or 'mbx'

    def get_forces(self, **kwargs):
        force = None
        return force

    def get_energy(self, olevel = 0, **kwargs) -> float:
        if olevel == 0 :
            energy = mbx.get_energy_pbc_nograd(self.positions, len(self.labels), self.box, units = 'au')
            e2 = mbx.get_external_field_contribution_to_energy(units = 'au')
            sprint('mbx -> energies', energy, e2, energy - e2, comm = self.comm)
            energy = energy - e2
        else :
            energy = 0.0
        return energy

    @print2file()
    def initial(self, inputfile = 'mbx.json', comm=None, **kwargs):
        self.comm = comm or self.comm
        inputs = self.write_input(**kwargs)
        self.positions = inputs['positions']
        self.monomer_natoms = inputs['monomer_natoms']
        self.labels = inputs['labels']
        self.monomer_names = inputs['monomer_names']
        self.inputfile = inputfile
        self.box = inputs['box']
        self.zval = inputs['zval']
        # mbx.initialize_system(self.positions, self.monomer_natoms, self.labels, self.monomer_names, self.inputfile, units = 'au')
        #-----------------------------------------------------------------------
        monomer_names = self.monomer_names.copy()
        for i in range(len(monomer_names)):
            if self.xc == 'pbe' :
                if monomer_names[i] == 'h2o' :
                    monomer_names[i] = 'mbpbe'
            elif self.xc == 'mbx' :
                pass
            else :
                raise AttributeError(f"Sorry, MBX only support 'MBX' and 'PBE' xc, not {self.xc}.")
        mbx.initialize_system(self.positions, self.monomer_natoms, self.labels, monomer_names, self.inputfile, units = 'au')
        #-----------------------------------------------------------------------
        self.npoints = len(self.labels)
        for name in self.monomer_names :
            if name == 'h2o' : self.npoints += 1
            """
            H2O : O H H M
                dipole : O H H 0
                charge : 0 H H M
            """
        # #-----------------------------------------------------------------------
        self.points_mm = mbx.get_xyz(self.npoints, units = 'au')
        self.points_mm = np.array(self.points_mm).reshape((-1, 3))
        self.points_zval = None

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
        mbx.initialize_system(xyz,number_of_atoms_per_monomer,atom_names,monomer_names,json_file, units = 'au')
        mbx.set_coordinates(xyz, natom, units = 'au')
        mbx.set_box(box, units = 'au')
        #-----------------------------------------------------------------------
        self.subcell = subcell
        self.npoints = nmol*4
        #-----------------------------------------------------------------------
        self.points_mm = mbx.get_xyz(self.npoints, units = 'au')
        self.points_mm = np.array(self.points_mm).reshape((-1, 3))/subcell.grid.latparas

    def get_points_zval(self, values = None):
        if self.points_zval is None :
            if values is None : values = self.zval
            zval = []
            ia = 0
            for im, name in enumerate(self.monomer_names) :
                if name == 'h2o' :
                    zval.append(0)
                    for k in ['H', 'H', 'O'] :
                        zval.append(values[k])
                    ia += 3
                else :
                    for k in self.labels[ia:ia + self.monomer_natoms[im]] :
                        zval.append(values[k])
                    ia += self.monomer_natoms[im]
            self.points_zval = np.array(zval)
        return self.points_zval

    def get_m_sites(self):
        index = []
        index_m = []
        pos = []
        ia = 0
        for im, name in enumerate(self.monomer_names) :
            if name == 'h2o' :
                index.append(ia)
                ia += 3
                index_m.append(ia + len(index_m))
            else :
                ia += self.monomer_natoms[im]
        return self.points_mm[index_m], index_m, index

    @print2file()
    def update_ions(self, subcell=None, **kwargs) :
        mbx.set_coordinates(self.positions, len(self.labels), units = 'au')
        mbx.set_box(self.box, units = 'au')

    def scf(self, **kwargs):
        pass

    def get_potential(self, out = None, grid = None, **kwargs):
        grid_points = grid.r.reshape((3, -1)).T
        grid_points = grid_points.ravel()
        pot, potfield = mbx.get_potential_and_electric_field_on_points(grid_points, grid_points.size//3, units = 'au')
        return pot.reshape(grid.nrR)

    def set_extpot(self, extpot = None, **kwargs):
        if self.comm.rank > 0 : return
        pot = self.get_value_at_points(extpot, self.points_mm).ravel()
        extfield = extpot.gradient()
        # extfield = extpot.gradient(flag = 'standard')
        # extfield = extpot.gradient(flag = 'supersmooth')
        potfield = self.get_value_at_points(extfield, self.points_mm).ravel()
        # potfield = []
        # for i in range(3):
            # p = self.get_value_at_points(extfield[i], self.points_mm)
            # potfield.append(p)
        # potfield= np.asarray(potfield).T.ravel()
        mbx.set_potential_and_electric_field_on_sites(pot, potfield, units = 'au')

    def get_value_at_points(self, data, points):
        if data.ndim > 3 :
            values = np.empty((points.shape[0], data.shape[0]))
        else :
            values = np.empty(points.shape[0])
        for i, p in enumerate(points):
            ip = (np.rint(p*data.grid.nrR/data.grid.latparas[0:3])).astype('int32')
            if data.ndim > 3 :
                values[i] = data[:, ip[0], ip[1], ip[2]]
            else :
                values[i] = data[ip[0], ip[1], ip[2]]
        return values

    def get_grid(self, gaps = 2.0, **kwargs):
        nr = (self.subcell.grid.latparas / gaps).astype('int32')
        self.nr = nr
        return nr

    def get_charges(self, **kwargs):
        charges = mbx.get_charges(self.npoints, units = 'au')
        positions = self.points_mm
        return charges, positions

    def get_dipoles(self, **kwargs):
        self.get_energy(**kwargs)
        dipoles = mbx.get_induced_dipoles(self.npoints, units = 'au')
        dipoles = np.array(dipoles).reshape((-1, 3))
        positions = self.points_mm
        return dipoles, positions

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
        inputs['zval'] = subcell.ions.Zval
        if inputs.get('box', None) is None :
            inputs['box'] = subcell.ions.pos.cell.lattice.ravel()/self.units['length']

        self.subcell = subcell

        if self.comm.rank > 0 :
            inputs['labels'] = ['Li']
            inputs['positions'] = [0.0, 0.0, 0.0]
            inputs['monomer_names'] = ['li+']
            inputs['monomer_natoms'] = [1]
            inputs['zval'] = {'Li' : 3.0}
            inputs['box'] = [10, 0, 0, 0, 10, 0, 0, 0, 10]
        return inputs
