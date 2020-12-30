import numpy as np
import time
import copy
from dftpy.formats import io
from dftpy.ewald import ewald
# from .utils.common import Field
from edftpy.mpi import sprint
from edftpy.properties import get_total_forces, get_total_stress
from edftpy.hartree import Hartree


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

    def get_energy(self, totalrho = None, totalfunc = None, update = None, olevel = 0, **kwargs):
        elist = []
        if totalfunc is None :
            if totalrho is None :
                totalrho = self.gsystem.density.copy()
            totalfunc = self.gsystem.total_evaluator(totalrho, calcType = ['E'])
        ene = totalfunc.energy / totalrho.mp.size
        elist.append(ene)
        for i, driver in enumerate(self.drivers):
            if driver is None :
                ene = 0.0
            elif olevel < 0 : # use saved energy
                ene = driver.energy
            elif update is not None and not update[i]:
                ene = driver.energy
            else :
                self.gsystem.density[:] = totalrho
                driver(density =driver.density, gsystem = self.gsystem, calcType = ['E'], olevel = olevel)
                ene = driver.energy
            elist.append(ene)
        elist = np.asarray(elist)
        etotal = np.sum(elist) + self.gsystem.ewald.energy
        etotal = self.gsystem.grid.mp.asum(etotal)
        elist = self.gsystem.grid.mp.vsum(elist)
        # print('elist', etotal, elist)
        return (etotal, elist)

    def update_density(self, update = None, **kwargs):
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
        sprint('Begin optimize')
        if gsystem is None:
            if self.gsystem is None :
                raise AttributeError("Must provide global system")
            gsystem = self.gsystem
        else:
            self.gsystem = gsystem

        self.gsystem.gaussian_density[:] = 0.0
        self.gsystem.density[:] = 0.0
        for i, driver in enumerate(self.drivers):
            if driver is None :
                density = None
                gaussian_density = None
            else :
                density = driver.density
                gaussian_density = driver.gaussian_density
            self.gsystem.update_density(gaussian_density, isub = i, fake = True)
            self.gsystem.update_density(density, isub = i)
        sprint('update density')
        sprint('density', self.gsystem.density.integral())
        totalrho = self.gsystem.density.copy()
        # io.write('a.xsf', totalrho, ions = self.gsystem.ions)
        # exit(0)
        self.nsub = len(self.drivers)
        energy_history = [0.0]
        #-----------------------------------------------------------------------
        fmt = "           {:8s}{:24s}{:16s}{:10s}{:10s}{:16s}".format("Step", "Energy(a.u.)", "dE", "dP", "dC", "Time(s)")
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
        of_ids = []
        of_drivers = []
        for i, driver in enumerate(self.drivers):
            if driver is not None and driver.technique== 'OF' :
                    of_drivers.append(driver)
                    of_ids.append(i)
        #-----------------------------------------------------------------------
        update = [True for _ in range(self.nsub)]
        res_norm = np.ones(self.nsub)
        totalrho_prev = None
        for it in range(self.options['maxiter']):
            self.iter = it
            #-----------------------------------------------------------------------
            if self.nsub == len(of_drivers):
                pass
            else :
                self.gsystem.total_evaluator.get_embed_potential(totalrho, gaussian_density = self.gsystem.gaussian_density, with_global = True)
                static_potential = self.gsystem.total_evaluator.static_potential.copy()
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

                update[i] = self.get_update(driver, it, res_norm[i])

                if update[i] :
                    self.gsystem.density[:] = totalrho
                    driver(gsystem = self.gsystem, calcType = ['O'], olevel = olevel)

            if self.nsub == len(of_drivers):
                static_potential = self.gsystem.total_evaluator.static_potential.copy()
            #-----------------------------------------------------------------------
            # res_norm = self.get_diff_residual()
            totalrho, totalrho_prev = totalrho_prev, totalrho
            totalrho = self.update_density(update = update)
            res_norm = self.get_diff_residual()
            d_ehart = self.delta_rho_hartree(totalrho, totalrho_prev)
            if self.check_converge(d_ehart, res_norm):
                sprint("#### Subsytem Density Optimization Converged ####")
                break
            scf_correction = self.get_scf_correction(static_potential, totalrho, totalrho_prev)
            totalfunc = self.gsystem.total_evaluator(totalrho, calcType = ['E'])
            energy = self.get_energy(totalrho, totalfunc, update = update, olevel = olevel)[0]
            #-----------------------------------------------------------------------
            energy_history.append(energy)
            timecost = time.time()
            dE = energy_history[-1] - energy_history[-2]
            dE += scf_correction
            energy += scf_correction
            #-----------------------------------------------------------------------
            fmt = "    Embed: {:<8d}{:<24.12E}{:<16.6E}{:<10.2E}{:<10.2E}{:<16.6E}".format(it, energy, dE, d_ehart, scf_correction, timecost - time_begin)
            fmt += "\n    Total: {:<8d}{:<24.12E}".format(it, totalfunc.energy)
            sprint(seq +'\n' + fmt +'\n' + seq)
            # if self.check_converge_ene(energy_history, res_norm):
                # sprint("#### Subsytem Density Optimization Converged ####")
                # break
            # exit()
        totalfunc = self.gsystem.total_evaluator(totalrho, calcType = ['E'])
        self.energy = self.get_energy(totalrho, totalfunc, olevel = 0)[0]
        self.density = totalrho
        fmt = "    Final: {:<8d}{:<24.12E}{:<16.6E}{:<10.2E}{:<10.2E}{:<16.6E}".format(it, self.energy, 0.0, d_ehart, 0.0, time.time() - time_begin)
        fmt += "\n    Total: {:<8d}{:<24.12E}".format(it, totalfunc.energy)
        sprint(seq +'\n' + fmt +'\n' + seq)
        return

    def get_update(self, driver, istep, residual):
        update_delay = driver.options['update_delay']
        update_freq = driver.options['update_freq']
        if istep > update_delay and (istep - update_delay) % update_freq > 0:
            update = False
        # elif residual < 1E-12 :
            # #For driver safe (e.g. pwscf will stop writing davcio)
            # update = False
        else :
            update = True
        return update

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
        diff_res = np.zeros(self.nsub)
        for i, driver in enumerate(self.drivers):
            if driver is not None :
                diff_res[i] = driver.residual_norm
        diff_res = self.gsystem.grid.mp.vsum(diff_res)
        sprint('diff_res', diff_res)
        return diff_res

    def check_converge(self, de, residual = None, **kwargs):
        econv = self.options["econv"]/1E2
        if np.any(residual < 1E-12) : return True
        #-----------------------------------------------------------------------
        if econv is not None :
            if abs(de) > econv : return False
        return True

    def check_converge_ene(self, energy_history, residual = None, **kwargs):
        econv = self.options["econv"]
        ncheck = self.options["ncheck"]
        E = energy_history[-1]
        #-----------------------------------------------------------------------
        # res_max = max(residual)
        # if res_max < 1E-12 : return True
        if np.any(residual < 1E-12) : return True
        #-----------------------------------------------------------------------
        if econv is not None :
            if len(energy_history) < ncheck + 1 :
                return False
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

    def get_forces(self, **kwargs):
        forces = get_total_forces(drivers = self.drivers, gsystem = self.gsystem, **kwargs)
        return forces

    def get_stress(self, **kwargs):
        stress = get_total_stress(drivers = self.drivers, gsystem = self.gsystem, **kwargs)
        return stress

    def get_scf_correction(self, v, rho1, rho0):
        ene = -np.sum((rho1 - rho0) * v) * rho1.grid.dV
        ene = self.gsystem.grid.mp.asum(ene)
        return ene

    def delta_rho_hartree(self, rho1=None, rho0=None):
        if rho0 is None or rho1 is None : return 1.0
        ene = Hartree.compute(rho1-rho0, calcType=['E']).energy
        ene = self.gsystem.grid.mp.asum(ene)
        return ene
