import numpy as np
from scipy import signal
from abc import ABC, abstractmethod
import os

from dftpy.constants import ENERGY_CONV

from edftpy.mixer import PulayMixer, AbstractMixer
from edftpy.utils.common import Grid, Field, Functional
from edftpy.utils.math import grid_map_data
from edftpy.utils import clean_variables
from edftpy.functional import hartree_energy
from edftpy.mpi import sprint, MP
from edftpy.engine.engine import Driver, Engine


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
            self.mixer = PulayMixer(predtype = 'kerker', predcoef = [1.0, 0.6, 1.0], maxm = 7, coef = 0.5, predecut = 0, delay = 1)
        elif isinstance(self.mixer, float):
            self.mix_coef = self.mixer

        fstr = f'Subcell grid({self.prefix}): {self.subcell.grid.nrR}  {self.subcell.grid.nr}\n'
        fstr += f'Subcell shift({self.prefix}): {self.subcell.grid.shift}\n'
        if self.grid_driver is not None :
            fstr += f'{self.__class__.__name__} has two grids :{self.grid.nrR} and {self.grid_driver.nrR}'
        sprint(fstr, comm=self.comm, level=1)
        self.engine.write_stdout(fstr)
        #-----------------------------------------------------------------------
        self.update_workspace(first = True, restart = self.restart)

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
        if isinstance(self.mixer, AbstractMixer):
            self.mixer.restart()
        if subcell is not None :
            self.subcell = subcell

        self.gaussian_density = self.get_gaussian_density(self.subcell, grid = self.grid)

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
            core_charge = self.atmp
        self.engine.get_rho_core(core_charge)
        self.core_density= self._format_field_invert(core_charge)

        self.core_density_sub = Field(grid = self.grid_sub)
        self.grid_sub.scatter(self.core_density, out = self.core_density_sub)
        if self.comm.rank == 0 :
            clean_variables(core_charge)
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
        if self._grid_sub is None :
            mp = MP(comm = self.comm, decomposition = self.subcell.grid.mp.decomposition)
            self._grid_sub = Grid(self.subcell.grid.lattice, self.subcell.grid.nrR, direct = True, mp = mp)
        return self._grid_sub

    def get_grid_driver(self, grid):
        nr = np.zeros(3, dtype = 'int32')
        self.engine.get_grid(nr)
        if not np.all(grid.nrR == nr) and self.comm.rank == 0 :
            grid_driver = Grid(grid.lattice, nr, direct = True)
        else :
            grid_driver = None
        return grid_driver

    def _driver_initialise(self, append = False, **kwargs):
        self.engine.set_stdout(self.outfile, append = append)
        if self.task == 'optical' :
            self.engine.tddft_initial(**kwargs)
        else :
            self.engine.initial(**kwargs)

        self.grid_driver = self.get_grid_driver(self.grid)
        self.init_density(**kwargs)

    def init_density(self, rho_ini = None, density_initial = None, **kwargs):
        if self.grid_driver is not None :
            grid = self.grid_driver
        else :
            grid = self.grid

        if self.comm.rank == 0 :
            self.density = Field(grid=self.grid)
            self.prev_density = Field(grid=self.grid)
            self.charge = np.empty((grid.nnr, self.nspin), order = self.engine.units['order'])
            self.prev_charge = np.empty((grid.nnr, self.nspin), order = self.engine.units['order'])
        else :
            self.density = self.atmp
            self.prev_density = self.atmp
            self.charge = self.atmp2
            self.prev_charge = self.atmp2

        self.density_sub = Field(grid = self.grid_sub)
        self.gaussian_density_sub = Field(grid = self.grid_sub)

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
        if density is None :
            density = self.density

        if grid is None :
            grid = self.grid_driver

        if self.grid_driver is not None and self.comm.rank == 0:
            charge = grid_map_data(density, grid = grid)
        else :
            charge = density
        charge = charge.reshape((-1, self.nspin), order = self.engine.units['order']) / self.engine.units['volume']
        return charge

    def _format_field_invert(self, charge = None, grid = None, **kwargs):
        if self.comm.rank > 0 : return self.atmp
        if charge is None :
            charge = self.charge

        if grid is None :
            grid = self.grid

        if self.grid_driver is not None and np.any(self.grid_driver.nrR != grid.nrR):
            density = Field(grid=self.grid_driver, direct=True, data = charge, order = self.engine.units['order'])
            rho = grid_map_data(density, grid = grid)
        else :
            rho = Field(grid=grid, rank=1, direct=True, data = charge, order = self.engine.units['order'])
        rho *= self.engine.units['volume']
        return rho

    def _get_extpot_serial(self, with_global = True, **kwargs):
        if self.comm.rank == 0 :
            self.evaluator.get_embed_potential(self.density, gaussian_density = self.gaussian_density, with_global = with_global)
            extpot = self.evaluator.embed_potential
        else :
            extpot = self.atmp
        return extpot

    def _get_extpot_mpi(self, with_global = True, **kwargs):
        self.grid_sub.scatter(self.density, out = self.density_sub)
        if self.gaussian_density is not None :
            self.grid_sub.scatter(self.gaussian_density, out = self.gaussian_density_sub)
        self.evaluator.get_embed_potential(self.density_sub, gaussian_density = self.gaussian_density_sub, gather = True, with_global = with_global)
        extpot = self.evaluator.embed_potential
        return extpot

    def get_extpot(self, extpot = None, mapping = True, **kwargs):
        if extpot is None :
            # extpot = self._get_extpot_serial(**kwargs)
            extpot = self._get_extpot_mpi(**kwargs)
        if mapping :
            if self.grid_driver is not None and self.comm.rank == 0:
                extpot = grid_map_data(extpot, grid = self.grid_driver)
            extpot = extpot.ravel(order = self.engine.units['order'])
        return extpot

    def _get_extene(self, extpot, **kwargs):
        if self.comm.rank == 0 :
            extene = (extpot * self.density).integral()
        else :
            extene = 0.0

        if self.comm.size > 1 :
            extene = self.comm.bcast(extene, root = 0)
        return extene

    def _get_charge(self, **kwargs):
        if self.task == 'optical' :
            self.engine.tddft(**kwargs)
        else :
            self.engine.scf(**kwargs)

        self.engine.get_rho(self.charge)
        return self.charge

    def get_density(self, vext = None, sdft = 'sdft', **kwargs):
        self._iter += 1
        if self._iter == 1 :
            # The first step density mixing also need keep initial = True
            initial = True
        else :
            initial = False

        self.prev_density[:] = self.density

        if self.mix_coef is not None :
            # If use pwscf mixer, do not need format the density
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
        #
        self._get_charge(initial = initial)
        #
        self.energy = 0.0
        self.dp_norm = 1.0
        self.density[:] = self._format_field_invert()
        return self.density

    def get_energy(self, olevel = 0, sdft = 'sdft', **kwargs):
        if olevel == 0 :
            # Here, we directly use saved density
            if sdft == 'pdft' :
                extpot = self.evaluator.embed_potential
                extpot = self.get_extpot(extpot, mapping = True)
            else :
                extpot = self.get_extpot()
            self.engine.set_extpot(extpot)
            energy = self.engine.calc_energy()
        else :
            energy = 0.0
        return energy

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
            sprint(fstr, comm=self.comm, level=1)
            self.engine.write_stdout(fstr)
            self.energy = func.energy
        return func

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
            sprint(fstr, comm=self.comm, level=1)
            self.engine.write_stdout(fstr)
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

        if self.comm.rank > 0 :
            self.residual_norm = 0.0
            self.dp_norm = 0.0
        # if self.comm.size > 1 : self.residual_norm = self.comm.bcast(self.residual_norm, root=0)
        return self.density

    def get_fermi_level(self, **kwargs):
        results = self.engine.get_ef()
        return results

    def get_forces(self, icalc = 3, **kwargs):
        """
        icalc :
            0 : all                              : 000
            1 : no ewald                         : 001
            2 : no local                         : 010
            3 : no ewald and local               : 011
        """
        forces = self.engine.get_force(icalc = icalc, **kwargs)
        return forces

    def get_stress(self, **kwargs):
        pass

    def end_scf(self, **kwargs):
        if self.task == 'optical' :
            self.engine.end_tddft()
        else :
            self.engine.end_scf()

    def save(self, save = ['D'], **kwargs):
        self.engine.save(save)

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
    def __init__(self, engine = None, **kwargs):
        kwargs["technique"] = kwargs.get("technique", 'EX')
        super().__init__(**kwargs)

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

    def init_density(self, rho_ini = None, density_initial = None, **kwargs):
        pass

    def get_energy(self, olevel = 0, **kwargs):
        if olevel == 0 :
            energy = self.engine.calc_energy()
        else :
            energy = 0.0
        return energy

    def get_energy_potential(self, density = None, calcType = ['E', 'V'], olevel = 1, **kwargs):
        func = Functional(name = 'ZERO', energy=0.0, potential=None)
        # self.engine.set_extpot(self.evaluator.global_potential, **kwargs)
        if 'V' in calcType :
            self._iter += 1
            rho = self._format_field(density)
            self.engine.scf(rho, **kwargs)
            pot = self.engine.get_potential(**kwargs)
            func.potential = self._format_field_invert(pot)
        if 'E' in calcType :
            energy = self.get_energy(olevel = olevel)
            func.energy = energy
            if self.comm.rank > 0 : func.energy = 0.0
            fstr = f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm, level=1)
            self.engine.write_stdout(fstr)
        return func

class DriverMM(DriverEX):
    """
    Note :
        The potential and density will gather in rank == 0 for engine.
    """
    def __init__(self, **kwargs):
        kwargs["technique"] = kwargs.get("technique", 'MM')
        super().__init__(**kwargs)

    def _driver_initialise(self, append = False, **kwargs):
        self.engine.initial(filename = self.filename, comm = self.comm,
                subcell = self.subcell, grid = self.grid)

    def get_energy_potential(self, density = None, calcType = ['E', 'V'], olevel = 1, **kwargs):
        func = Functional(name = 'ZERO', energy=0.0, potential=None)
        self.engine.set_extpot(self.evaluator.global_potential, **kwargs)
        if 'V' in calcType :
            pot = self.engine.get_potential(grid = self.grid_driver, **kwargs)
            func.potential = self._format_field_invert(pot)
        if 'E' in calcType :
            energy = self.get_energy(olevel = olevel)
            func.energy = energy
            if self.comm.rank > 0 : func.energy = 0.0
            fstr = f'sub_energy({self.prefix}): {self._iter}  {func.energy}'
            sprint(fstr, comm=self.comm, level=1)
            self.engine.write_stdout(fstr)
        return func

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
