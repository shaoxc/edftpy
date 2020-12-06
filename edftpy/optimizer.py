import numpy as np
import time
import copy
from dftpy.formats import io
from dftpy.ewald import ewald
# from .utils.common import Field
from edftpy.mpi import sprint


class Optimization(object):

    def __init__(self, drivers=None, gsystem = None, options=None):
        if drivers is None:
            raise AttributeError("Must provide optimization driver (list)")
        else:
            self.drivers = drivers

        self.gsystem = gsystem

        default_options = {
            "maxiter": 80,
            "econv": 1.0e-6,
            "ncheck": 2,
        }

        self.options = default_options
        self.options.update(options)

    def get_energy(self, func_list = None, **kwargs):
        energy = 0.0
        if func_list is not None :
            for func in func_list :
                if func is not None :
                    energy += func.energy
        energy += self.gsystem.ewald.energy
        energy = self.gsystem.grid.mp.asum(energy)
        return energy

    def get_energy_all(self, totalrho = None, totalfunc = None, olevel = 0, **kwargs):
        elist = []
        if totalfunc is None :
            if totalrho is None :
                totalrho = self.gsystem.density.copy()
            totalfunc = self.gsystem.total_evaluator(totalrho, calcType = ['E'])
        elist.append(totalfunc.energy)
        for i, driver in enumerate(self.drivers):
            if driver is None :
                ene = 0.0
            else :
                self.gsystem.density[:] = totalrho
                driver(density =driver.density, gsystem = self.gsystem, calcType = ['E'], olevel = olevel)
                ene = driver.energy
            elist.append(ene)
        elist = np.asarray(elist)
        etotal = np.sum(elist) + self.gsystem.ewald.energy
        etotal = self.gsystem.grid.mp.asum(etotal)
        energy = [etotal, elist]
        return energy

    def update_density(self, update = None, mu = None, **kwargs):
        #-----------------------------------------------------------------------
        # diff_res = self.get_diff_residual()
        # sprint('diff_res', diff_res)
        for i, driver in enumerate(self.drivers):
            # coef = driver.calculator.mixer.coef.copy()
            # sprint('coef',coef, 'i', i)
            # coef[0] = self.get_frag_coef(coef[0], diff_res[i], diff_res)
            # sprint('outcoef',coef)
            if driver is None : continue
            coef = None
            if update[i] :
                driver.update_density(coef = coef)

        for i, driver in enumerate(self.drivers):
            if driver is None :
                density = None
            else :
                density = driver.density
            if i == 0 :
                self.gsystem.update_density(density, isub = i, restart = True)
            else :
                self.gsystem.update_density(density, isub = i, restart = False)
        totalrho = self.gsystem.density.copy()
        return totalrho

    def optimize(self, gsystem = None, olevel = 1, **kwargs):
        #-----------------------------------------------------------------------
        if gsystem is None:
            if self.gsystem is None :
                raise AttributeError("Must provide global system")
            gsystem = self.gsystem
        else:
            self.gsystem = gsystem

        self.gsystem.gaussian_density[:] = 0.0
        for i, driver in enumerate(self.drivers):
            if driver is None :
                density = None
                gaussian_density = None
            else :
                density = driver.density
                gaussian_density = driver.gaussian_density
                # if gaussian_density is not None :
                    # print('gaussian_density : -> ', gaussian_density.integral())
            self.gsystem.update_density(gaussian_density, isub = i, restart = False, fake = True)
            if i == 0 :
                self.gsystem.update_density(density, isub = i, restart = True)
            else :
                self.gsystem.update_density(density, isub = i, restart = False)
        totalrho = self.gsystem.density.copy()
        # value = self.gsystem.gaussian_density.gather()
        # if self.gsystem.graphtopo.rank == 0 :
            # io.write('a.xsf', value, ions = self.gsystem.ions)
        # exit(0)
        self.nsub = len(self.drivers)
        energy_history = []

        mu_list = np.zeros(self.nsub)
        func_list = [None,] * (self.nsub + 1)
        energy_history = [0.0]
        #-----------------------------------------------------------------------
        fmt = "           {:8s}{:24s}{:16s}{:16s}{:8s}{:16s}".format("Step", "Energy(a.u.)", "dE", "dP", "Nls", "Time(s)")
        resN = 9999
        #-----------------------------------------------------------------------
        sprint(fmt)
        seq = "-" * 100
        time_begin = time.time()
        timecost = time.time()
        # resN = np.einsum("..., ...->", residual, residual, optimize = 'optimal') * rho.grid.dV
        # dE = energy
        # fmt = "    Embed: {:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<16.6E}".format(0, energy, dE, resN, 1, timecost - time_begin)
        # sprint(seq +'\n' + fmt +'\n' + seq)
        #-----------------------------------------------------------------------
        embed_keys = []
        of_ids = []
        of_drivers = []
        for i, driver in enumerate(self.drivers):
            if driver is not None :
                if driver.technique== 'OF' :
                    of_drivers.append(driver)
                    of_ids.append(i)
                else :
                    embed_keys = driver.evaluator.embed_evaluator.funcdicts.keys()
                    break
        # print('embed_keys', embed_keys)
        # exit()
        #-----------------------------------------------------------------------
        update = [True for _ in range(self.nsub)]
        for it in range(self.options['maxiter']):
            self.iter = it
            #-----------------------------------------------------------------------
            if self.nsub == len(of_drivers):
                pass
            else :
                self.gsystem.total_evaluator.get_embed_potential(totalrho, gaussian_density = self.gsystem.gaussian_density, embed_keys = embed_keys, with_global = True)
            for isub in range(self.nsub + len(of_drivers)):
                if isub < self.nsub :
                    driver = self.drivers[isub]
                    # initial the global_potential
                    if driver is None :
                        global_potential = None
                    elif driver.technique == 'OF' :
                        continue
                    else :
                        if driver.evaluator.global_potential is None :
                            driver.evaluator.global_potential = np.zeros(driver.grid.nrR)
                        global_potential = driver.evaluator.global_potential
                    self.gsystem.sub_value(self.gsystem.total_evaluator.embed_potential, global_potential, isub = isub)
            #-----------------------------------------------------------------------
            for isub in range(self.nsub + len(of_drivers)):
                if isub < self.nsub :
                    driver = self.drivers[isub]
                    if driver is None or driver.technique == 'OF' :
                        continue
                    i = isub
                else :
                    driver = of_drivers[isub - self.nsub]
                    i = of_ids[isub - self.nsub]

                update_delay = driver.options['update_delay']
                update_freq = driver.options['update_freq']
                if it > update_delay and (it - update_delay) % update_freq > 0:
                    update[i] = False
                else :
                    update[i] = True

                if update[i] :
                    self.gsystem.density[:] = totalrho
                    driver(gsystem = self.gsystem, calcType = ['O', 'E'], olevel = olevel)
                    func_list[i] = driver.functional
                    mu_list[i] = driver.mu
            # res_norm = self.get_diff_residual()
            #-----------------------------------------------------------------------
            totalrho = self.update_density(mu = mu_list, update = update)
            res_norm = self.get_diff_residual()

            totalfunc = self.gsystem.total_evaluator(totalrho, calcType = ['E'])
            func_list[i + 1] = totalfunc
            energy = self.get_energy(func_list)
            #-----------------------------------------------------------------------
            energy_history.append(energy)
            timecost = time.time()
            dE = energy_history[-1] - energy_history[-2]
            resN = max(res_norm)
            fmt = "    Embed: {:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<16.6E}".format(it, energy, dE, resN, 1, timecost - time_begin)
            fmt += "\n    Total: {:<8d}{:<24.12E}".format(it, totalfunc.energy)
            sprint(seq +'\n' + fmt +'\n' + seq)
            if self.check_converge(energy_history):
                sprint("#### Subsytem Density Optimization Converged ####")
                break
        if self.options['maxiter'] < 1 :
            totalfunc = self.gsystem.total_evaluator(totalrho, calcType = ['E'])
        # self.energy = energy
        self.energy = self.get_energy_all(totalrho, totalfunc, olevel = 0)[0]
        self.density = totalrho
        return

    def get_frag_coef(self, coef, sub_d_e, d_e, alpha = 1.0, maxs = 0.2):
        '''
        '''
        d_e = np.array(d_e)
        d_min = np.min(d_e)
        d_max = np.max(d_e)
        sub_d_e = abs(sub_d_e)
        if d_max > 1E-10 :
            # new_coef = alpha * (coef * (1.0 - sub_d_e/d_max)) + coef * (1.0 - alpha)
            new_coef = (1.0 - sub_d_e * (sub_d_e - d_min) / d_max ** 2) * coef
            new_coef = max(new_coef, maxs * coef)
        else :
            new_coef = coef
        return new_coef

    def get_diff_residual(self, **kwargs):
        diff_res = []
        for i, driver in enumerate(self.drivers):
            if driver is not None :
                diff_res.append(driver.residual_norm)
        return diff_res

    def check_converge(self, energy_history, **kwargs):
        econv = self.options["econv"]
        ncheck = self.options["ncheck"]
        E = energy_history[-1]
        if econv is not None :
            if len(energy_history) - 1 < ncheck :
                return
            for i in range(ncheck):
                dE = abs(energy_history[-2-i] - E)
                if abs(dE) > econv :
                    return False
        return True

    def _update_try(self, it, res_norm):
        update = [True for _ in range(len(self.drivers))]
        freq = 5
        for i, item in enumerate(res_norm):
            if i == 0 :
                ide = 1
                if it > freq and (it - freq) % freq > 0 :
                    update[i] = False
                # if it > freq and it < 150 and (it - freq) % freq > 0 :
                # if it > freq :
                    # if abs(res_history[i][-1]-res_history[i][-2]) < 1E-15 and \
                            # res_history[ide][-1] < 1E-6 and res_history[ide][-1]-res_history[ide][-2] < 2E-7 :
                        # update[i] = True
                    # else :
                        # update[i] = False
                else :
                    update[i] = True

                if update[i] :
                    if it > freq :
                        self.drivers[ide].calculator.mixer.restart()
                        self.drivers[ide].calculator.mixer._delay = 0
        return update

    def _transfer_electron(self, denlist = None, mu = None, **kwargs):
        #-----------------------------------------------------------------------
        scale = 0.05
        mu = np.array(mu)
        mu_av = np.mean(mu)
        d_mu = mu - mu_av

        Ni = np.zeros_like(mu)
        for i, rho_in in enumerate(denlist):
            Ni[i] = rho_in.integral()
        d_Ni = d_mu / mu * Ni
        d_Ni -= np.sum(d_Ni)/len(d_Ni)
        sprint('Ni', Ni)
        sprint('d_Ni', d_Ni)
        if np.max(np.abs(d_Ni)) > 1E-2 :
            Ni_new = Ni + d_Ni * scale
            sprint('Ni_new', Ni_new)
            for i, rho_in in enumerate(denlist):
                rho_in *= Ni_new[i]/Ni[i]

    def get_diff_residual_ene(self, **kwargs):
        diff_res = []
        for i, driver in enumerate(self.drivers):
            if driver is not None :
                r = driver.density - driver.prev_density
                res_norm = np.sqrt(driver.grid.mp.asum(r * r)/driver.grid.nnrR)
                diff_res.append(res_norm)
            # energy = Hartree.compute(driver.density.copy(), calcType=['E']).energy
            # sprint('Hartree_i', i, energy)
            # ls = driver.get_energy_traj('HARTREE', density = driver.density)
            # if len(ls) > 1 :
                # dr = abs(ls[-1]-ls[-2])
            # else :
                # dr = 0.0
            # diff_res.append(dr)
        return diff_res
