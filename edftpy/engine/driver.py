import numpy as np

from dftpy.constants import ENERGY_CONV

from edftpy.mixer import Mixer
from edftpy.utils.common import Grid, Field, Functional
from edftpy.utils.math import grid_map_data
from edftpy.functional import hartree_energy
from edftpy.mpi import sprint, MP
from edftpy.engine.engine import Driver
from edftpy.io import print2file
from edftpy.density import build_pseudo_density


class DriverConstraint(object):
    """
    Give some constraint for the driver.

    Notes:
        Most time still use driver to do the thing.
    """
    # funcdicts = ['get_density', 'update_density', 'end_scf',
    #         'density', 'driver', 'residual_norm', 'dp_norm']

    def __init__(self, driver = None, density = None, **kwargs):
        self.driver = driver
        self.density = density
        if density is None :
            self.density = self.driver.density.copy()
        self.residual_norm = 0.0
        self.dp_norm = 0.0

    def get_density(self, **kwargs):
        return self.density

    def update_density(self, **kwargs):
        return self.density

    def end_scf(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        pass

    def __getattr__(self, attr):
        if attr in dir(self):
            return object.__getattribute__(self, attr)
        else :
            return getattr(self.driver, attr)

class DriverKS(Driver):
    """
    Note :
        The potential and density will gather in rank == 0 for engine.
    """
    def __init__(self, engine = None, **kwargs):
        kwargs["technique"] = kwargs.get("technique", 'KS')
        super().__init__(**kwargs)
        self.engine = engine

        self._driver_initialise(**kwargs)

        if self.mixer is None :
            self.mixer = Mixer(scheme = 'pulay', predtype = 'kerker', predcoef = [1.0, 0.6, 1.0], maxm = 7, coef = 0.5, predecut = 0, delay = 1)
        elif isinstance(self.mixer, (int, float)):
            self.mix_coef = self.mixer

        fstr = f'Subcell grid({self.prefix}): {self.subcell.grid.nrR}  {self.subcell.grid.nr}\n'
        fstr += f'Subcell shift({self.prefix}): {self.subcell.grid.shift}\n'
        if self.grid_driver is not None :
            fstr += f'{self.prefix} has two grids :{self.grid.nrR} and {self.grid_driver.nrR}\n'
        else :
            fstr += f'{self.prefix} has same grids :{self.grid.nrR} and {self.grid.nrR}\n'
        self.write_stdout(fstr)
        # sprint(fstr, comm = self.comm)
        #-----------------------------------------------------------------------
        self.update_workspace(first = True, restart = self.restart)

    @print2file()
    def update_workspace(self, subcell = None, first = False, update = 0, restart = False, **kwargs):
        """
        Notes:
            clean workspace
        """
        self.rho = None
        self.wfs = None
        self.occupations = None
        self.eigs = None
        self.fermi = None
        self.calc = None
        self._iter = 0
        self.energy = 0.0
        self.phi = None
        self.residual_norm = 1.0
        self.dp_norm = 1.0
        if hasattr(self.mixer, 'restart') : self.mixer.restart()
        if subcell is not None : self.subcell = subcell

        if not first :
            self.engine.update_ions(self.subcell, update)
            if update == 0 :
                # get new density
                self.engine.get_rho(self.charge)
                self.density[:] = self._format_field_invert()
        if self.task == 'optical' :
            if first :
                self.engine.tddft_after_scf()
                if restart :
                    self.engine.wfc2rho()
                    # get new density
                    self.engine.get_rho(self.charge)
                    self.density[:] = self._format_field_invert()

        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        if self.comm.rank == 0 :
            core_charge = np.empty(grid.nnr, order = self.engine.units['order'])
        else :
            core_charge = self.atmp2
        self.engine.get_rho_core(core_charge)
        self.core_density_sub = self.subcell.core_density

        # Use engine core_density
        self.core_density = self._format_field_invert(core_charge)
        self.grid_sub.scatter(self.core_density, out = self.core_density_sub)
        #Use eDFTpy core density------------------------------------------------
        # self.core_density = self.core_density_sub.gather(grid = self.grid)
        #-----------------------------------------------------------------------
        if self.comm.rank == 0 :
            fstr = f'ncharge({self.prefix}): {self._iter} {self.density.integral()}'
            sprint(fstr, comm = self.comm)

        if self.subcell.gaussian_density is None :
            self.subcell.gaussian_density = self.subcell.core_density
        self.gaussian_density = self.get_gaussian_density(self.subcell, grid = self.grid)
        #
        self.density_sub = self.subcell.density
        self.gaussian_density_sub = self.subcell.gaussian_density
        return

    @property
    def grid(self):
        if self._grid is None :
            if np.all(self.subcell.grid.nrR == self.subcell.grid.nr):
                self._grid = self.subcell.grid
            else :
                self._grid = Grid(self.subcell.grid.lattice, self.subcell.grid.nrR, direct = True)
        return self._grid

    @property
    def grid_sub(self):
        return self.subcell.grid

    @print2file()
    def get_grid_driver(self, grid):
        nr = self.engine.get_grid()
        if not np.all(grid.nrR == nr) and self.comm.rank == 0 :
            grid_driver = Grid(grid.lattice, nr, direct = True)
        else :
            grid_driver = None
        return grid_driver

    @print2file()
    def _driver_initialise(self, **kwargs):
        if self.task == 'optical' :
            self.engine.tddft_initial(**kwargs)
        else :
            self.engine.initial(**kwargs)

        self.grid_driver = self.get_grid_driver(self.grid)
        self.init_density(**kwargs)

    @print2file()
    def init_density(self, rho_ini = None, density_initial = None, **kwargs):
        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        if self.comm.rank == 0 :
            self.density = Field(grid=self.grid, rank=self.nspin)
            self.prev_density = Field(grid=self.grid, rank=self.nspin)
            self.charge = np.empty((grid.nnr, self.nspin), order = self.engine.units['order'])
            self.prev_charge = np.empty((grid.nnr, self.nspin), order = self.engine.units['order'])
        else :
            self.density = self.atmp2
            self.prev_density = self.atmp2
            self.charge = self.atmp2
            self.prev_charge = self.atmp2

        if rho_ini is None :
            if density_initial and density_initial != 'temp' :
                rho_ini = self.subcell.density.gather(grid = self.grid)

        if rho_ini is not None :
            if self.comm.rank == 0 : self.density[:] = rho_ini
            self.charge[:] = self._format_field()
            self.engine.set_rho(self.charge)
        else :
            self.engine.get_rho(self.charge)
            self.density[:] = self._format_field_invert()

    def _format_field(self, density = None, grid = None, **kwargs):
        if density is None : density = self.density

        if grid is None and self.comm.rank == 0 :
            grid = self.grid_driver if self.grid_driver is not None else density.grid

        if self.comm.rank == 0 and np.any(density.grid.nrR != grid.nrR):
            charge = grid_map_data(density, grid = grid)
        else :
            charge = density

        if self.comm.rank == 0 :
            nspin = density.shape[0] if density.ndim == 4 else 1
            if nspin > 1 :
                value = np.empty((grid.nnrR, nspin), order = self.engine.units['order'])
                for i in range(nspin):
                    value[:, i] = charge[i].ravel(order = self.engine.units['order']) / self.engine.units['volume']
            else :
                value = charge.reshape((-1, nspin), order = self.engine.units['order']) / self.engine.units['volume']
        else :
            value = self.atmp2
        return value

    def _format_field_invert(self, charge = None, grid = None, **kwargs):
        if charge is None : charge = self.charge
        if self.comm.rank > 0 : return charge

        if grid is None : grid = self.grid

        nspin = charge.shape[-1] if charge.ndim == 2 else 1
        if self.grid_driver is not None and np.any(self.grid_driver.nrR != grid.nrR):
            density = Field(grid=self.grid_driver, direct=True, data = charge.ravel(order='C'), order = self.engine.units['order'], rank=nspin)
            rho = grid_map_data(density, grid = grid)
        else :
            rho = Field(grid=grid, direct=True, data = charge.ravel(order='C'), order = self.engine.units['order'], rank=nspin)
        rho *= self.engine.units['volume']
        return rho

    @print2file()
    def _get_extpot_serial(self, with_global = True, **kwargs):
        if self.comm.rank == 0 :
            self.evaluator.get_embed_potential(self.density, gaussian_density = self.gaussian_density, with_global = with_global)
            extpot = self.evaluator.embed_potential
        else :
            extpot = self.atmp2
        return extpot

    @print2file()
    def _get_extpot_mpi(self, with_global = True, **kwargs):
        self.grid_sub.scatter(self.density, out = self.density_sub)
        self.evaluator.get_embed_potential(self.density_sub, gaussian_density = self.gaussian_density_sub, gather = True, with_global = with_global)
        extpot = self.evaluator.embed_potential
        return extpot

    @print2file()
    def get_extpot(self, extpot = None, mapping = True, **kwargs):
        if extpot is None :
            # extpot = self._get_extpot_serial(**kwargs)
            extpot = self._get_extpot_mpi(**kwargs)
        if mapping :
            extpot = self._format_field(extpot) / self.engine.units['energy']
        return extpot

    @print2file()
    def _get_extene(self, extpot, **kwargs):
        if self.comm.rank == 0 :
            extene = (extpot * self.density).integral()
        else :
            extene = 0.0

        if self.comm.size > 1 :
            extene = self.comm.bcast(extene, root = 0)
        return extene

    @print2file()
    def _get_charge(self, **kwargs):
        if self.task == 'optical' :
            self.engine.tddft(**kwargs)
        else :
            self.engine.scf(**kwargs)

        self.engine.get_rho(self.charge)
        return self.charge

    @print2file()
    def _get_density_prep(self, sdft = 'sdft', **kwargs):
        self._iter += 1

        self.prev_density[:] = self.density

        if self.mix_coef is not None :
            # If use engine mixer, do not need format the density
            pass
        else :
            self.charge[:] = self._format_field()
            self.engine.set_rho(self.charge)

        if sdft == 'pdft' :
            extpot = self.evaluator.embed_potential
            extpot = self.get_extpot(extpot, mapping = True)
        else :
            extpot = self.get_extpot()

        self.prev_charge[:] = self.charge

        self.engine.set_extpot(extpot)

    @print2file()
    def get_density(self, sdft = 'sdft', occupations = None, sum_band = False, **kwargs):
        if sum_band or occupations is not None :
            self.engine.sum_band(occupations=occupations, **kwargs)
            self.engine.get_rho(self.charge)
        elif sdft == 'scdft' :
            return self.get_bands(sdft=sdft, **kwargs)
        else :
            self._get_density_prep(sdft=sdft, **kwargs)
            self._get_charge()
        self.energy = 0.0
        self.dp_norm = 1.0
        self.density[:] = self._format_field_invert()
        if self.comm.rank == 0 :
            fstr = f'ncharge({self.prefix}): {self._iter} {self.density.integral()}'
            sprint(fstr, comm = self.comm)
            # from edftpy.io import write
            # write(self.prefix+'.xsf', self.density, ions = self.subcell.ions)
        return self.density

    @print2file()
    def get_bands(self, sdft = 'sdft', sum_band = False, **kwargs):
        self._get_density_prep(sdft=sdft, **kwargs)
        self.engine.scf(sum_band = sum_band, **kwargs)
        self.band_energies = self.engine.get_band_energies(**kwargs).reshape((self.nspin, -1))
        self.band_weights = self.engine.get_band_weights(**kwargs).reshape((self.nspin, -1))
        return self.band_energies, self.band_weights

    @print2file()
    def get_energy(self, olevel = 0, sdft = 'sdft', **kwargs):
        if olevel == 0 :
            # Here, we directly use saved density
            if sdft == 'pdft' :
                extpot = self.evaluator.embed_potential
                extpot = self.get_extpot(extpot, mapping = True)
            else :
                extpot = self.get_extpot()
            self.engine.set_extpot(extpot)
            energy = self.engine.get_energy(olevel = olevel) * self.engine.units['energy']
        else :
            energy = 0.0
        return energy

    @print2file()
    def get_energy_potential(self, density = None, calcType = ['E', 'V'], olevel = 1, sdft = 'sdft', **kwargs):
        if 'E' in calcType :
            energy = self.get_energy(olevel = olevel, sdft = sdft)

            if olevel == 0 :
                self.grid_sub.scatter(density, out = self.density_sub)
                edict = self.evaluator(self.density_sub, calcType = ['E'], with_global = False, with_embed = False, gather = True, split = True)
                func = edict.pop('TOTAL')
            else : # elif olevel == 1 :
                if self.comm.rank == 0 :
                    func = self.evaluator(density, calcType = ['E'], with_global = False, with_embed = True)
                else :
                    func = Functional(name = 'ZERO', energy=0.0, potential=None)

            func.energy += energy
            if sdft == 'sdft' and self.exttype == 0 : func.energy = 0.0
            if self.comm.rank > 0 : func.energy = 0.0

            if olevel == 0 :
                fstr = format("Energy information", "-^80") + '\n'
                for key, item in edict.items():
                    fstr += "{:>12s} energy: {:22.15E} (eV) = {:22.15E} (a.u.)\n".format(key, item.energy* ENERGY_CONV["Hartree"]["eV"], item.energy)
                fstr += "-" * 80 + '\n'
            else :
                fstr = ''
            fstr += f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            # self.write_stdout(fstr)
            sprint(fstr, comm = self.comm)
            self.energy = func.energy
        return func

    @print2file()
    def update_density(self, coef = None, mix_grid = False, **kwargs):
        # mix_grid = True
        if self.comm.rank == 0 :
            if self.grid_driver is not None and mix_grid:
                prev_density = self._format_field_invert(self.prev_charge, self.grid_driver)
                density = self._format_field_invert(self.charge, self.grid_driver)
            else :
                prev_density = self.prev_density
                density = self.density
            #-----------------------------------------------------------------------
            r = density - prev_density
            self.residual_norm = np.sqrt(np.sum(r * r)/r.size)
            rmax = r.amax()
            fstr = f'res_norm({self.prefix}): {self._iter}  {rmax}  {self.residual_norm}'
            # self.write_stdout(fstr)
            sprint(fstr, comm = self.comm)
            #-----------------------------------------------------------------------
            if self.mix_coef is None :
                self.dp_norm = hartree_energy(r)
                rho = self.mixer(prev_density, density, **kwargs)
                if self.grid_driver is not None and mix_grid:
                    rho = grid_map_data(rho, grid = self.grid)
                self.density[:] = rho

        if self.mix_coef is not None :
            if coef is None : coef = self.mix_coef
            self.engine.scf_mix(coef = coef)
            self.dp_norm = self.engine.get_dnorm()
            self.engine.get_rho(self.charge)
            self.density[:] = self._format_field_invert()
        else :
            if self.comm.size > 1 : self.dp_norm = self.comm.bcast(self.dp_norm, root=0)
            self.set_dnorm(self.dp_norm)

        if self.comm.rank > 0 :
            self.residual_norm = 0.0
            self.dp_norm = 0.0
        # if self.comm.size > 1 : self.residual_norm = self.comm.bcast(self.residual_norm, root=0)
        return self.density

    def set_dnorm(self, dnorm, **kwargs):
        self.dp_norm = dnorm
        self.engine.set_dnorm(self.dp_norm)
        if self.comm.rank > 0 :
            self.dp_norm = 0.0

    @print2file()
    def get_fermi_level(self, **kwargs):
        results = self.engine.get_ef()
        return results

    @print2file()
    def get_forces(self, icalc = 3, **kwargs):
        """
        icalc :
            0 : all                              : 000
            1 : no ewald                         : 001
            2 : no local                         : 010
            3 : no ewald and local               : 011
        """
        forces = self.engine.get_forces(icalc = icalc, **kwargs)
        forces *= self.engine.units['energy']/self.engine.units['length']
        return forces

    @print2file()
    def get_stress(self, **kwargs):
        pass

    @print2file()
    def end_scf(self, **kwargs):
        if self.task == 'optical' :
            self.engine.end_tddft()
        else :
            self.engine.end_scf()

    @print2file()
    def save(self, save = ['D'], **kwargs):
        self.engine.save(save)

    @print2file()
    def stop_run(self, status = 0, save = ['D'], **kwargs):
        if self.task == 'optical' :
            self.engine.stop_tddft(status, save = save)
        else :
            self.engine.stop_scf(status, save = save)

class DriverEX(DriverKS):
    """
    Note :
        The potential and density will gather in rank == 0 for engine.
    """
    def __init__(self, **kwargs):
        kwargs["technique"] = kwargs.get("technique", 'EX')
        super().__init__(**kwargs)

    @print2file()
    def update_workspace(self, subcell = None, first = False, update = 0, restart = False, **kwargs):
        """
        Notes:
            clean workspace
        """
        self.fermi = None
        self._iter = 0
        self.energy = 0.0
        self.residual_norm = 0.0
        self.dp_norm = 0.0

        if not first :
            self.engine.update_ions(self.subcell, update = update)
        return

    # @print2file()
    # def init_density(self, rho_ini = None, density_initial = None, **kwargs):
        # pass

    @print2file()
    def get_energy(self, olevel = 0, **kwargs):
        if olevel == 0 :
            energy = self.engine.get_energy(olevel = olevel) * self.engine.units['energy']
        else :
            energy = 0.0
        return energy

    @print2file()
    def update_density(self, **kwargs):
        return self.density

    @print2file()
    def get_energy_potential(self, density = None, calcType = ['E', 'V'], olevel = 1, **kwargs):
        func = Functional(name = 'ZERO', energy=0.0, potential=None)
        # self.engine.set_extpot(self.evaluator.global_potential, **kwargs)
        if 'V' in calcType :
            self._iter += 1
            rho = self._format_field(density)
            self.engine.scf(rho, **kwargs)
            pot = self.engine.get_potential(**kwargs) * self.engine.units['energy']
            func.potential = self._format_field_invert(pot)
        if 'E' in calcType :
            energy = self.engine.get_energy(olevel = olevel) * self.engine.units['energy']
            func.energy = energy
            if self.comm.rank > 0 : func.energy = 0.0
            fstr = f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            # self.write_stdout(fstr)
            sprint(fstr, comm = self.comm)
        return func

class DriverMM(DriverKS):
    """
    Note :
        The potential and density will gather in rank == 0 for engine.
    """
    def __init__(self, **kwargs):
        kwargs["technique"] = kwargs.get("technique", 'MM')
        super().__init__(**kwargs)

    @print2file()
    def update_workspace(self, subcell = None, first = False, update = 0, restart = False, **kwargs):
        """
        Notes:
            clean workspace
        """
        self.fermi = None
        self._iter = 0
        self.energy = 0.0
        self.residual_norm = 0.0
        self.dp_norm = 0.0

        if not first :
            self.engine.update_ions(self.subcell, update = update)
        return

    def _driver_initialise(self, append = False, **kwargs):
        self.engine.initial(filename = self.filename, comm = self.comm,
                subcell = self.subcell, grid = self.grid)
        self.grid_driver = self.get_grid_driver(self.grid)
        self.init_density(**kwargs)

    @print2file()
    def init_density(self, rho_ini = None, density_initial = None, sigma = 0.6, rcut = 10.0, **kwargs):
        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        if self.comm.rank == 0 :
            self.density = Field(grid=self.grid, rank=self.nspin)
            self.prev_density = Field(grid=self.grid, rank=self.nspin)
            self.charge = np.empty((grid.nnr, self.nspin), order = self.engine.units['order'])
            self.prev_charge = np.empty((grid.nnr, self.nspin), order = self.engine.units['order'])
        else :
            self.density = self.atmp2
            self.prev_density = self.atmp2
            self.charge = self.atmp2
            self.prev_charge = self.atmp2

        self.density_charge_sub = Field(grid = self.grid_sub, rank=self.nspin)

        charges, positions_c = self.engine.get_charges()
        charges = self.engine.get_points_zval() - charges
        positions_c = positions_c*self.engine.units['length']
        #-----------------------------------------------------------------------
        if self.comm.size > 1 :
            charges = self.comm.bcast(charges, root = 0)
            positions_c = self.comm.bcast(positions_c, root = 0)
        #-----------------------------------------------------------------------
        self.density_charge_sub[:] = 0.0
        for c, p in zip(charges, positions_c):
            if c > 1 :
                sigma2 = 1.4 *sigma
            else :
                sigma2 = sigma
            self.density_charge_sub = build_pseudo_density(p, self.grid_sub, scale = c, sigma = sigma2, rcut = rcut,
                    density = self.density_charge_sub, add = True, deriv = 0)
        #Double density---------------------------------------------------------
        self.density_charge_mo_sub = Field(grid = self.grid_sub, rank=self.nspin)
        self.density_charge_mo_sub[:] = 0.0
        pos_m, inds_m, inds_o = self.engine.get_m_sites()
        if len(inds_m) > 0 :
            dipoles, positions_d = self.engine.get_dipoles()
            if self.comm.size > 1 :
                positions_d = self.comm.bcast(positions_d, root = 0)
            positions_c[inds_m] = positions_d[inds_o]
            #
            for c, p in zip(charges, positions_c):
                if c > 1 :
                    sigma2 = 1.4 *sigma
                else :
                    sigma2 = sigma
                self.density_charge_mo_sub = build_pseudo_density(p, self.grid_sub, scale = c, sigma = sigma2, rcut = rcut,
                        density = self.density_charge_mo_sub, add = True, deriv = 0)
        else :
            self.density_charge_mo_sub = self.density_charge_sub
        #-----------------------------------------------------------------------
        sprint('charges :\n', charges, comm = self.comm)
        # self.density_charge_sub.write('1_pseudo_density_charge.xsf', ions = self.subcell.ions)
        self.density_sub = self.subcell.density
        self.gaussian_density_sub = self.subcell.gaussian_density
        self.core_density_sub = self.subcell.core_density
        self.core_density = self.core_density_sub.gather(grid = self.grid)
        self.density_charge = self.density_charge_sub.gather(grid = self.grid)
        self.density_charge_mo = self.density_charge_mo_sub.gather(grid = self.grid)

    @print2file()
    def get_energy(self, olevel = 0, **kwargs):
        if olevel == 0 :
            energy = self.engine.get_energy(olevel = olevel) * self.engine.units['energy']
        else :
            energy = 0.0
        return energy

    @print2file()
    def update_density(self, **kwargs):
        return self.density

    @print2file()
    def get_energy_potential(self, density = None, calcType = ['E', 'V'], olevel = 1, **kwargs):
        func = Functional(name = 'ZERO', energy=0.0, potential=None)
        if olevel == 0 :
            self.engine.set_extpot(self.evaluator.global_potential / self.engine.units['energy'], **kwargs)
        # if 'V' in calcType :
            # pot = self.engine.get_potential(grid = self.grid_driver, **kwargs) * self.engine.units['energy']
            # func.potential = self._format_field_invert(pot)
        if 'E' in calcType :
            energy = self.engine.get_energy(olevel = olevel) * self.engine.units['energy']
            func.energy = energy
            if self.comm.rank > 0 : func.energy = 0.0
            fstr = f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            # self.write_stdout(fstr)
            sprint(fstr, comm = self.comm)
        return func

    @print2file()
    def get_density(self, rcut = 10, sigma = 1.7, **kwargs):
        #
        #-----------------------------------------------------------------------
        # if self.comm.rank == 0 :
            # from edftpy.io import read_density
            # self.density[:] = read_density('sub_mbx_0.xsf')
            # self.density_charge[:] = self.density
        # return self.density
        #-----------------------------------------------------------------------
        self.engine.set_extpot(self.evaluator.global_potential / self.engine.units['energy'], **kwargs)
        #
        dipoles, positions_d = self.engine.get_dipoles()
        sprint('dipoles0 :\n', dipoles, comm = self.comm)
        positions_d = positions_d*self.engine.units['length']
        dipoles = dipoles*self.engine.units['length']
        #-----------------------------------------------------------------------
        if self.comm.size > 1 :
            dipoles = self.comm.bcast(dipoles, root = 0)
            positions_d = self.comm.bcast(positions_d, root = 0)
        # dip = np.loadtxt('edftpy_mm_dipole.txt').reshape((-1,3))
        # dipoles[:3] = dip[:3]
        # dipoles[:] = 0.0
        sprint('dipoles :\n', dipoles, comm = self.comm)
        #-----------------------------------------------------------------------
        self.density_sub[:] = 0.0
        # self.density_sub[:] = self.density_charge_sub
        for c, p in zip(dipoles, positions_d):
            self.density_sub = build_pseudo_density(p, self.grid_sub, scale = c, sigma = sigma, rcut = rcut,
                    density = self.density_sub, add = True, deriv = 1)
        self.density_sub.gather(out = self.density, root = 0)
        # self.density_sub.write('1_density_charge.xsf', ions = self.subcell.ions)
        return self.density

    @print2file()
    def get_density_v0(self, rcut = 10, sigma = 0.6, **kwargs):
        charges, positions_c = self.engine.get_charges()
        charges = self.engine.get_points_zval() - charges
        dipoles, positions_d = self.engine.get_dipoles()
        #-----------------------------------------------------------------------
        positions_c = positions_c*self.engine.units['length']
        positions_d = positions_d*self.engine.units['length']
        dipoles = dipoles*self.engine.units['length']
        #-----------------------------------------------------------------------
        self.density[:] = 0.0
        for c, p in zip(charges, positions_c):
            if c > 1 :
                sigma2 = 1.4 *sigma
            else :
                sigma2 = sigma
            self.density = build_pseudo_density(p, self.grid, scale = c, sigma = sigma2, rcut = rcut,
                    density = self.density, add = True, deriv = 0)
        for c, p in zip(dipoles, positions_d):
            self.density = build_pseudo_density(p, self.grid, scale = c, sigma = sigma, rcut = rcut,
                    density = self.density, add = True, deriv = 1)
        #
        sprint('charges :\n', charges, comm = self.comm)
        sprint('dipoles :\n', dipoles, comm = self.comm)
        return self.density


class DriverOF:
    def __init__(self, engine = None, **kwargs):
        if engine is None :
            from edftpy.engine.engine_dftpy import EngineDFTpy
            engine = EngineDFTpy(**kwargs)
        self.engine = engine

    def __getattr__(self, attr):
        if attr == 'engine' :
            return object.__getattribute__(self, attr)
        else :
            return getattr(self.engine, attr)

    def __call__(self, *args, **kwargs):
        self.engine(*args, **kwargs)
