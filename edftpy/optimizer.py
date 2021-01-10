import numpy as np
import time
from dftpy.formats import io
from dftpy.constants import ENERGY_CONV
from edftpy.mpi import sprint
from edftpy.properties import get_total_forces, get_total_stress
from collections import OrderedDict
import pprint


class Optimization(object):

    def __init__(self, drivers=None, gsystem = None, options=None):
        if drivers is None:
            raise AttributeError("Must provide optimization driver (list)")

        self.drivers = drivers
        self.gsystem = gsystem

        default_options = {
            "maxiter": 80,
            "econv": 1.0e-5,
            "pconv": None,
            "pconv_sub": None,
            "ncheck": 2,
        }

        self.options = default_options
        self.options.update(options)
        self.nsub = len(self.drivers)

    def guess_pconv(self):
        if not self.options['pconv'] :
            self.options['pconv'] = self.options['econv'] / 1E2

        if not self.options['pconv_sub'] :
            pconv_sub = np.zeros(self.nsub)
            for i, driver in enumerate(self.drivers):
                if driver is not None and driver.comm.rank == 0 :
                    pconv_sub[i] = self.options['pconv'] / driver.subcell.ions.nat * self.gsystem.ions.nat
            self.options['pconv_sub'] = self.gsystem.grid.mp.vsum(pconv_sub)
        pretty_dict_str = 'Optimization options :\n'
        pretty_dict_str += pprint.pformat(self.options)
        sprint(pretty_dict_str)

    def get_energy(self, totalrho = None, totalfunc = None, update = None, olevel = 0, **kwargs):
        elist = []
        if totalfunc is None :
            if totalrho is None :
                totalrho = self.gsystem.density.copy()
            totalfunc = self.gsystem.total_evaluator(totalrho, calcType = ['E'])
        elist.append(totalfunc.energy)
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
            if update is not None and update[i]:
                driver.update_density(coef = coef)

        self.gsystem.density[:] = 0.0
        for i, driver in enumerate(self.drivers):
            if driver is None :
                density = None
            else :
                density = driver.density
            self.gsystem.update_density(density, isub = i)
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

        self.guess_pconv()

        self.gsystem.gaussian_density[:] = 0.0
        self.gsystem.density[:] = 0.0
        for i, driver in enumerate(self.drivers):
            if driver is None :
                density = None
                gaussian_density = None
                core_density = None
            else :
                density = driver.density
                gaussian_density = driver.gaussian_density
                core_density = driver.core_density
                # io.write(str(i) + '_gauss.xsf', gaussian_density, ions = driver.subcell.ions)
            self.gsystem.update_density(density, isub = i)
            self.gsystem.update_density(gaussian_density, isub = i, fake = True)
            self.gsystem.update_density(core_density, isub = i, core = True)
        sprint('update density')
        sprint('density', self.gsystem.density.integral())
        totalrho = self.gsystem.density.copy()
        #-----------------------------------------------------------------------
        self.add_xc_correction()
        #-----------------------------------------------------------------------
        # io.write('a.xsf', totalrho, ions = self.gsystem.ions)
        # exit(0)
        energy_history = [0.0]
        #-----------------------------------------------------------------------
        fmt = "{:10s}{:8s}{:24s}{:16s}{:10s}{:10s}{:16s}".format(" ", "Step", "Energy(a.u.)", "dE", "dP", "dC", "Time(s)")
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
                            driver.evaluator.global_potential = np.zeros_like(driver.density)
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
            totalrho, totalrho_prev = totalrho_prev, totalrho
            totalrho = self.update_density(update = update)
            res_norm = self.get_diff_residual()
            dp_norm = self.get_diff_potential()
            sprint('diff_res', res_norm)
            sprint('dp_norm', dp_norm)
            if self.check_converge_potential(dp_norm):
                sprint("#### Subsytem Density Optimization Converged (Potential)####")
                break
            scf_correction = self.get_scf_correction(static_potential, totalrho, totalrho_prev)
            totalfunc = self.gsystem.total_evaluator(totalrho, calcType = ['E'])
            energy = self.get_energy(totalrho, totalfunc, update = update, olevel = olevel)[0]
            #-----------------------------------------------------------------------
            energy += scf_correction
            energy_history.append(energy)
            timecost = time.time()
            dE = energy_history[-1] - energy_history[-2]
            d_ehart = np.max(dp_norm)
            #-----------------------------------------------------------------------
            fmt = "{:>10s}{:<8d}{:<24.12E}{:<16.6E}{:<10.2E}{:<10.2E}{:<16.6E}".format("Embed: ", it, energy, dE, d_ehart, scf_correction, timecost - time_begin)
            fmt += "\n{:>10s}{:<8d}{:<24.12E}".format("Total: ", it, totalfunc.energy)
            sprint(seq +'\n' + fmt +'\n' + seq)
            if self.check_converge_energy(energy_history):
                sprint("#### Subsytem Density Optimization Converged (Energy)####")
                break
        self.energy_all = self.print_energy()
        self.energy = self.energy_all['TOTAL']
        self.density = totalrho
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
        return diff_res

    def get_diff_potential(self, **kwargs):
        dp_norm = np.zeros(self.nsub)
        for i, driver in enumerate(self.drivers):
            if driver is not None :
                dp_norm[i] = driver.dp_norm
        dp_norm = self.gsystem.grid.mp.vsum(dp_norm)
        return dp_norm

    def check_converge_potential(self, dp_norm, **kwargs):
        # pconv = self.options["pconv"]
        pconv = self.options["pconv_sub"]
        if np.any(dp_norm > pconv) : return False
        return True

    def check_converge_energy(self, energy_history, **kwargs):
        econv = self.options["econv"]
        ncheck = self.options["ncheck"]
        E = energy_history[-1]
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

    def print_energy(self):
        edict = self.gsystem.total_evaluator(self.gsystem.density, calcType = ['E'], split = True)
        totalfunc = edict.pop('TOTAL')
        totalfunc.energy *= self.gsystem.grid.mp.size
        totalfunc = self.gsystem.total_evaluator(self.gsystem.density, calcType = ['E'])
        etotal, elist = self.get_energy(self.gsystem.density, totalfunc, olevel = 0)
        keys = list(edict.keys())
        values = [item.energy for item in edict.values()]
        keys.append('II')
        values.append(self.gsystem.ewald.energy)
        values = self.gsystem.grid.mp.vsum(values)
        ep_w = OrderedDict()
        for i, key in sorted(enumerate(keys), key=lambda x:x[1]):
            ep_w[key] = values[i]
        for i, item in enumerate(elist[1:]):
            key = "SUB-"+str(i)
            ep_w[key] = item
        key = "TOTAL"
        ep_w[key] = etotal
        sprint(format("Energy information", "-^80"))
        for key, value in ep_w.items():
            sprint("{:>10s} energy: {:22.15E} (eV)  =  {:22.15E} (a.u.) ".format(key, value* ENERGY_CONV["Hartree"]["eV"], value))
        sprint("-" * 80)
        return ep_w

    def add_xc_correction(self):
        """
        Sorry, we also hate add the non-linear core correction here, maybe we can find a better way to add it in the future.
        """
        for i, driver in enumerate(self.drivers):
            if driver is None : continue
            if 'XC' in driver.evaluator.funcdicts :
                driver.evaluator.funcdicts['XC'].core_density = driver.core_density
            if driver.technique == 'OF' :
                if 'XC' in driver.evaluator_of.funcdicts :
                    driver.evaluator_of.funcdicts['XC'].core_density = driver.core_density
        if 'XC' in self.gsystem.total_evaluator.funcdicts :
            self.gsystem.total_evaluator.funcdicts['XC'].core_density = self.gsystem.core_density
