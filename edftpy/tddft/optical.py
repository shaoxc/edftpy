import numpy as np
import time
import os

from dftpy.constants import ENERGY_CONV

from edftpy.mpi import sprint
from edftpy.properties import get_dipole
from edftpy.optimizer import Optimization

class MoleculeOpticalAbsorption(Optimization):
    """
    The TDDFT class

    Notes:
        Not support NLKEDF.
    """

    def __init__(self, drivers=None, gsystem = None, options=None, optimizer = None, **kwargs):
        super().__init__(drivers=drivers, gsystem = gsystem, options=options)
        default_options = {
            "maxiter": 1000,
            "olevel": 2,
            "sdft": 'sdft',
            "restart" : 'initial',
            "maxtime" : 0,
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
                    driver.save(save = ['W', 'D'])
                    driver.task = 'optical'
                    driver.update_workspace(first = True)

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
        self.density, self.density_prev = self.density_prev, self.density
        self.density = self.gsystem.density.copy()

    def optimize(self, **kwargs):
        self.run(**kwargs)

    def run(self, **kwargs):
        for item in self.irun(**kwargs):
            pass
        return item

    def irun(self, restart = None, **kwargs):
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
            self.converged = False
            self.iter = self.options.get('iter', 0) # The iteration number of restart
            yield self.converged
            self.call_observers(self.iter)
        #-----------------------------------------------------------------------
        seq = "-" * 100
        self.converged = False
        for it in range(self.options['maxiter']):
            #-----------------------------------------------------------------------
            self.step()
            yield self.converged
            self.call_observers(self.iter)
            self.update_density()
            #-----------------------------------------------------------------------
            # use new density to calculate the energy is very important, otherwise the restart will not correct.
            self.energy = self.get_energy(density = self.density, update = self.update, olevel =self.options['olevel'])
            self.dip = get_dipole(self.density_prev, self.gsystem.ions)
            #-----------------------------------------------------------------------
            fmt = "{:>10s}{:<8d}{:<24.12E}{:<14.6E}{:<14.6E}{:<14.6E}{:<16.6E}".format(
                    "Tddft: ", self.iter, self.energy, *self.dip, time.time()- self.time_begin)
            sprint(seq +'\n' + fmt +'\n' + seq)
            if self.check_stop(): break
        else :
            self.converged = True
        yield self.converged
