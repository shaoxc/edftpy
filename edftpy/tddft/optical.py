import numpy as np
import time

from dftpy.constants import ENERGY_CONV

from edftpy.mpi import sprint
from edftpy.properties import get_dipole
from edftpy.optimizer import Optimization

class MoleculeOpticalAbsorption(Optimization):

    def __init__(self, drivers=None, gsystem = None, options=None, optimizer = None, **kwargs):
        super().__init__(drivers=drivers, gsystem = gsystem, options=options)
        default_options = {
            "maxiter": 1000,
            "olevel": 2,
            "sdft": 'sdft',
            "restart" : 'initial'
        }
        self.options = default_options
        if isinstance(options, dict):
            self.options.update(options)
        self.optimizer = optimizer
        self.iter = 0

    def initialization(self, restart = None):
        restart = restart or self.options.get('restart', 'initial')
        if restart == 'initial' :
            self.optimizer.optimize()
            for driver in self.drivers:
                if driver is not None :
                    driver.save()
                    driver.update_workspace(first = True)
                    driver.task = 'optical'

    def update_density(self, initial = False, **kwargs):
        self.gsystem.density[:] = 0.0
        if initial :
            self.gsystem.gaussian_density[:] = 0.0
            self.gsystem.core_density[:] = 0.0
        for i, driver in enumerate(self.drivers):
            if driver is None :
                density = None
                gaussian_density = None
                core_density = None
            else :
                density = driver.density
                if driver.gaussian_density is None :
                    driver.gaussian_density = driver.core_density
                gaussian_density = driver.gaussian_density
                core_density = driver.core_density
            self.gsystem.update_density(density, isub = i)
            if initial :
                self.gsystem.update_density(gaussian_density, isub = i, fake = True)
                self.gsystem.update_density(core_density, isub = i, core = True)

    def optimize(self, restart = None):
        self.initialization(restart = restart)
        self.update_density(initial = True)
        self.add_xc_correction()
        #-----------------------------------------------------------------------
        if self.iter == 0 :
            fmt = "{:10s}{:8s}{:24s}{:14s}{:14s}{:14s}{:16s}".format(" ", "Step", "Energy(a.u.)", "DIP(x)", "DIP(y)", "DIP(y)", "Time(s)")
            sprint(fmt)
            self.time_begin = time.time()
            self.density_prev = None
            self.density = self.gsystem.density.copy()
        #-----------------------------------------------------------------------
        olevel = self.options.get('olevel', 2)
        seq = "-" * 100
        sdft = self.options['sdft']
        for it in range(self.options['maxiter']):
            self.iter += 1
            # update the rhomax for NLKEDF
            self.set_kedf_params(level = 3) # If NL KEDF, always contains NL parts
            self.set_global_potential()
            for isub in range(self.nsub + len(self.of_drivers)):
                if isub < self.nsub :
                    driver = self.drivers[isub]
                    if driver is None or driver.technique == 'OF' :
                        continue
                else :
                    driver = self.of_drivers[isub - self.nsub]
                    isub = self.of_ids[isub - self.nsub]

                self.gsystem.density[:] = self.density
                driver(gsystem = self.gsystem, calcType = ['O'], olevel = olevel, sdft = sdft)
            #-----------------------------------------------------------------------
            self.density, self.density_prev = self.density_prev, self.density
            self.update_density()
            self.density = self.gsystem.density.copy()
            totalfunc = self.gsystem.total_evaluator(self.density, calcType = ['E'], olevel = olevel)
            self.energy = self.get_energy(self.density, totalfunc.energy, olevel = olevel)[0]
            self.dip = get_dipole(self.density, self.gsystem.ions)
            #-----------------------------------------------------------------------
            fmt = "{:>10s}{:<8d}{:<24.12E}{:<14.6E}{:<14.6E}{:<14.6E}{:<16.6E}".format(
                    "Embed: ", self.iter, self.energy, *self.dip, time.time()- self.time_begin)
            sprint(seq +'\n' + fmt +'\n' + seq)
        return

    def end_run(self, **kwargs):
        self.end_scf()
        self.energy_all = self.print_energy()
        self.energy = self.energy_all['TOTAL']
        return
