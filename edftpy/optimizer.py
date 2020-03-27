import numpy as np
import time
import copy

# from .utils import math as dftmath
from .mixer import LinearMixer


class Optimization(object):

    def __init__(self, opt_drivers=None, total_evaluator = None, options=None):

        if opt_drivers is None:
            raise AttributeError("Must provide optimization driver (list)")
        else:
            self.opt_drivers = opt_drivers

        if total_evaluator is None:
            raise AttributeError("Must provide total density evaluater")
        else:
            self.total_evaluator = total_evaluator

        default_options = {
            "maxfun": 50,
            "maxiter": 20,
            "maxls": 30,
            "econv": 1.0e-6,
        }

        self.options = default_options
        self.options.update(options)
        #-----------------------------------------------------------------------
        self.mixer = LinearMixer()
        #-----------------------------------------------------------------------

    def get_energy(self, func_list = None, **kwargs):
        energy = 0.0
        if func_list is not None :
            for func in func_list :
                energy += func.energy
        return energy

    def update_density(self, denlist = None, prev_denlist = None, **kwargs):
        for i, rho_in in enumerate(prev_denlist):
            denlist[i] = self.mixer(rho_in, denlist[i])

        totalrho = None
        for rho in denlist :
            if totalrho is None :
                totalrho = rho.copy()
            else :
                totalrho += rho
        return totalrho, denlist

    def optimize(self, guess_rho = None, **kwargs):
        energy_history = []

        prev_denlist = copy.deepcopy(guess_rho)
        totalrho = prev_denlist[0].copy()
        for item in prev_denlist[1:] :
            totalrho += item

        func_list = []
        for i, driver in enumerate(self.opt_drivers):
            driver(density = prev_denlist[i], rest_rho = totalrho - prev_denlist[i], calcType = ['E'])
            func_list.append(driver.functional)

        denlist = copy.deepcopy(prev_denlist)

        totalfunc = self.total_evaluator(totalrho)
        func_list.append(totalfunc)
        energy = self.get_energy(func_list)
        energy_history.append(energy)

        #-----------------------------------------------------------------------
        fmt = "           {:8s}{:24s}{:16s}{:16s}{:8s}{:16s}".format("Step", "Energy(a.u.)", "dE", "dP", "Nls", "Time(s)")
        resN = 9999
        #-----------------------------------------------------------------------
        print(fmt)
        time_begin = time.time()
        timecost = time.time()
        # resN = np.einsum("..., ...->", residual, residual, optimize = 'optimal') * rho.grid.dV
        dE = energy
        seq = "-" * 100
        fmt = "    Embed: {:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<16.6E}".format(0, energy, dE, resN, 1, timecost - time_begin)
        print(seq +'\n' + fmt +'\n' + seq)

        for it in range(self.options['maxiter']):
            prev_denlist, denlist = denlist, prev_denlist
            for i, driver in enumerate(self.opt_drivers):
                driver(density = prev_denlist[i], rest_rho = totalrho - prev_denlist[i], calcType = ['O', 'E'])
                denlist[i] = driver.density
                func_list[i] = driver.functional
            #-----------------------------------------------------------------------
            totalrho, denlist = self.update_density(denlist, prev_denlist)
            totalfunc = self.total_evaluator(totalrho)
            func_list[i + 1] = totalfunc
            energy = self.get_energy(func_list)
            #-----------------------------------------------------------------------
            energy_history.append(energy)
            timecost = time.time()
            dE = energy_history[-1] - energy_history[-2]
            fmt = "    Embed: {:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<16.6E}".format(it, energy, dE, resN, 1, timecost - time_begin)
            print(seq +'\n' + fmt +'\n' + seq)
            if abs(dE) < self.options["econv"]:
                if len(energy_history) > 2:
                    if abs(energy_history[-1] - energy_history[-3]) < self.options["econv"]:
                        print("#### Density Optimization Converged ####")
                        break
        self.energy = energy
        self.density = totalrho
        return
