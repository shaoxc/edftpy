import numpy as np
import time
import copy
from .hartree import Hartree


class Optimization(object):

    def __init__(self, opt_drivers=None, gsystem = None, options=None):

        if opt_drivers is None:
            raise AttributeError("Must provide optimization driver (list)")
        else:
            self.opt_drivers = opt_drivers

        self.gsystem = gsystem

        default_options = {
            "maxfun": 50,
            "maxiter": 20,
            "maxls": 30,
            "econv": 1.0e-6,
            "ncheck": 2,
        }

        self.options = default_options
        self.options.update(options)

    def get_energy(self, func_list = None, **kwargs):
        energy = 0.0
        elist = []
        if func_list is not None :
            for func in func_list :
                energy += func.energy
                elist.append(func.energy)
        elist.append(energy)
        return energy

    def update_density(self, denlist = None, prev_denlist = None, mu = None, update = None, **kwargs):
        #-----------------------------------------------------------------------
        diff_res = self.get_diff_residual()
        print('diff_res', diff_res)
        for i, driver in enumerate(self.opt_drivers):
            # coef = driver.calculator.mixer.coef.copy()
            # print('coef',coef, 'i', i)
            # coef[0] = self.get_frag_coef(coef[0], diff_res[i], diff_res)
            # print('outcoef',coef)
            coef = None
            if update[i] :
                denlist[i] = driver.update_density(coef = coef)
            else :
                denlist[i] = driver.density.copy()
        #-----------------------------------------------------------------------
        # if mu is not None :
        if 0 :
            print('mu', mu)
            scale = 0.05
            mu = np.array(mu)
            mu_av = np.mean(mu)
            d_mu = mu - mu_av

            Ni = np.zeros_like(mu)
            for i, rho_in in enumerate(denlist):
                Ni[i] = rho_in.integral()
            d_Ni = d_mu / mu * Ni
            d_Ni -= np.sum(d_Ni)/len(d_Ni)
            print('Ni', Ni)
            print('d_Ni', d_Ni)
            if np.max(np.abs(d_Ni)) > 1E-2 :
                Ni_new = Ni + d_Ni * scale
                print('Ni_new', Ni_new)
                for i, rho_in in enumerate(denlist):
                    rho_in *= Ni_new[i]/Ni[i]

        for i, item in enumerate(denlist):
            if i == 0 :
                self.gsystem.update_density(item, restart = True)
            else :
                self.gsystem.update_density(item, restart = False)
        totalrho = self.gsystem.density.copy()
        return totalrho, denlist

    def optimize(self, gsystem = None, guess_rho = None, **kwargs):
        #-----------------------------------------------------------------------
        if gsystem is None:
            if self.gsystem is None :
                raise AttributeError("Must provide global system")
        else:
            self.gsystem = gsystem
        #-----------------------------------------------------------------------
        self.nsub = len(self.opt_drivers)
        energy_history = []

        prev_denlist = copy.deepcopy(guess_rho)
        for i, item in enumerate(prev_denlist):
            if i == 0 :
                gsystem.update_density(item, restart = True)
            else :
                gsystem.update_density(item, restart = False)
        totalrho = gsystem.density.copy()

        mu_list = [[] for _ in range(len(self.opt_drivers))]
        func_list = [[] for _ in range(len(self.opt_drivers) + 1)]
        denlist = copy.deepcopy(prev_denlist)
        energy_history = [0.0]
        # for i, driver in enumerate(self.opt_drivers):
            # gsystem.density[:] = totalrho
            # driver(density = prev_denlist[i], gsystem = gsystem, calcType = ['E'])
            # func_list.append(driver.functional)
            # mu_list.append([])

        # totalfunc = self.gsystem.total_evaluator(totalrho, calcType = ['E'])
        # func_list.append(totalfunc)
        # energy = self.get_energy(func_list)
        # energy_history.append(energy)
        #-----------------------------------------------------------------------
        fmt = "           {:8s}{:24s}{:16s}{:16s}{:8s}{:16s}".format("Step", "Energy(a.u.)", "dE", "dP", "Nls", "Time(s)")
        resN = 9999
        #-----------------------------------------------------------------------
        print(fmt)
        seq = "-" * 100
        time_begin = time.time()
        timecost = time.time()
        # resN = np.einsum("..., ...->", residual, residual, optimize = 'optimal') * rho.grid.dV
        # dE = energy
        # fmt = "    Embed: {:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<16.6E}".format(0, energy, dE, resN, 1, timecost - time_begin)
        # print(seq +'\n' + fmt +'\n' + seq)

        update = [True for _ in range(len(self.opt_drivers))]
        for it in range(self.options['maxiter']):
            self.iter = it
            prev_denlist, denlist = denlist, prev_denlist
            for i, driver in enumerate(self.opt_drivers):
                #-----------------------------------------------------------------------
                update_delay = driver.options['update_delay']
                update_freq = driver.options['update_freq']
                if it > update_delay and (it - update_delay) % update_freq > 0:
                    update[i] = False
                else :
                    update[i] = True

                if update[i] :
                    gsystem.density[:] = totalrho
                    driver(density = prev_denlist[i], gsystem = gsystem, calcType = ['O', 'E'])
                    denlist[i] = driver.density
                    func_list[i] = driver.functional
                    mu_list[i] = driver.mu
            res_norm = self.get_diff_residual()
            #-----------------------------------------------------------------------
            if it > 3 :
                totalrho, denlist = self.update_density(denlist, prev_denlist, mu = mu_list, update = update)
            else :
                totalrho, denlist = self.update_density(denlist, prev_denlist, mu = None, update = update)

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
            print(seq +'\n' + fmt +'\n' + seq)
            if self.check_converge(energy_history):
                print("#### Subsytem Density Optimization Converged ####")
                break
        self.energy = energy
        self.density = totalrho
        self.subdens = denlist
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
        for i, driver in enumerate(self.opt_drivers):
            r = driver.density - driver.prev_density
            res_norm = float(np.sqrt(np.sum(r * r)/np.size(r)))
            diff_res.append(res_norm)
            # energy = Hartree.compute(driver.density.copy(), calcType=['E']).energy
            # print('Hartree_i', i, energy)
            # ls = driver.get_energy_traj('HARTREE', density = driver.density)
            # if len(ls) > 1 :
                # dr = abs(ls[-1]-ls[-2])
            # else :
                # dr = 0.0
            # diff_res.append(dr)
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
        update = [True for _ in range(len(self.opt_drivers))]
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
                        self.opt_drivers[ide].calculator.mixer.restart()
                        self.opt_drivers[ide].calculator.mixer._delay = 0
        return update
