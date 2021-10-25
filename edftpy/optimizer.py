import numpy as np
import time
from collections import OrderedDict
import pprint
import os

from dftpy.constants import ENERGY_CONV

from edftpy.mpi import sprint
from edftpy.properties import get_total_forces, get_total_stress
from edftpy.functional import hartree_energy
from edftpy.utils.common import Grid, Field, Functional
from edftpy.io import write
from edftpy.engine.driver import DriverConstraint


class Optimization(object):

    def __init__(self, drivers=None, gsystem = None, options=None):
        if drivers is None:
            raise AttributeError("Must provide optimization driver (list)")
        if gsystem is None :
            raise AttributeError("Must provide global system")

        self.drivers = drivers
        self.gsystem = gsystem

        default_options = {
            "maxiter": 80,
            "econv": 1.0e-5,
            "pconv": None,
            "pconv_sub": None,
            "ncheck": 2,
            "olevel": 2,
            "sdft": 'sdft',
            "maxtime" : 0,
        }

        self.options = default_options
        if isinstance(options, dict):
            self.options.update(options)
        self.nsub = len(self.drivers)
        self.of_drivers = []
        self.of_ids = []
        for i, driver in enumerate(self.drivers):
            if driver is not None and driver.technique== 'OF' :
                    self.of_drivers.append(driver)
                    self.of_ids.append(i)
        self.energy = 0
        self.iter = 0
        self.density = None
        self.density_prev = None
        self.converged = False
        self.update = [True for _ in range(self.nsub)]
        self.observers = []

    def guess_pconv(self):
        if self.options['pconv'] is None:
            self.options['pconv'] = self.options['econv'] / 1E2

        if self.options['pconv_sub'] is None :
            pconv_sub = np.zeros(self.nsub)
            for i, driver in enumerate(self.drivers):
                if driver is not None and driver.comm.rank == 0 :
                    pconv_sub[i] = self.options['pconv'] * driver.subcell.ions.nat / self.gsystem.ions.nat
            self.options['pconv_sub'] = self.gsystem.grid.mp.vsum(pconv_sub)
        pretty_dict_str = 'Optimization options :\n'
        pretty_dict_str += pprint.pformat(self.options)
        sprint(pretty_dict_str)

    def get_energy(self, totalrho = None, total_energy= None, update = None, olevel = 0, **kwargs):
        elist = []
        if total_energy is None :
            if totalrho is None :
                totalrho = self.gsystem.density.copy()
            total_energy = self.gsystem.total_evaluator(totalrho, calcType = ['E'], olevel = olevel).energy
        elist.append(total_energy)
        for i, driver in enumerate(self.drivers):
            if driver is None :
                ene = 0.0
            elif olevel < 0 : # use saved energy
                ene = driver.energy
            elif update is not None and not update[i]:
                ene = driver.energy
            else :
                self.gsystem.density[:] = totalrho
                driver(density =driver.density, gsystem = self.gsystem, calcType = ['E'], olevel = olevel, sdft = self.options['sdft'])
                ene = driver.energy
            elist.append(ene)
        elist = np.asarray(elist)
        etotal = np.sum(elist) + self.gsystem.ewald.energy
        etotal = self.gsystem.grid.mp.asum(etotal)
        elist = self.gsystem.grid.mp.vsum(elist)
        # print('elist', etotal, elist)
        return (etotal, elist)

    def update_density(self, update = None, **kwargs):
        # diff_res = self.get_diff_residual()
        for i, driver in enumerate(self.drivers):
            if driver is None : continue
            # coef = driver.calculator.mixer.coef.copy()
            # coef[0] = self.get_frag_coef(coef[0], diff_res[i], diff_res)
            coef = None
            if update is not None and update[i]:
                driver.update_density(coef = coef)

        self.gsystem.density[:] = 0.0
        for i, driver in enumerate(self.drivers):
            root = 0
            if driver is None :
                density = None
                potential = None
            else :
                density = driver.density
                potential = driver.potential
                if driver.comm.rank == 0 : root = self.gsystem.comm.rank
            root = self.gsystem.grid.mp.amax(root)
            technique = self._get_driver_technique(driver)
            if technique in ['EX', 'MM'] :
                extpots = self.gsystem.total_evaluator.external_potential
                pot = extpots.get(technique, None)
                if pot is None :
                    pot = Field(grid=self.gsystem.grid)
                    extpots[technique] = pot
                if potential is None and driver is not None:
                    if driver.comm.rank == 0 :
                        potential = Field(grid=driver.grid)*0.0
                    else :
                        potential = driver.atmp
                self.gsystem.grid.scatter(potential, out = pot, root = root)
            else :
                self.gsystem.update_density(density, isub = i)
        self.density, self.density_prev = self.density_prev, self.density
        self.density = self.gsystem.density.copy()
        return

    def step(self, **kwargs):
        # Update the rhomax for NLKEDF, first step without NL will be better for scf not for tddft.
        self.set_kedf_params()
        self.set_global_potential()
        for isub in range(self.nsub + len(self.of_drivers)):
            if isub < self.nsub :
                driver = self.drivers[isub]
                if driver is None or driver.technique == 'OF' : continue
            else :
                driver = self.of_drivers[isub - self.nsub]
                isub = self.of_ids[isub - self.nsub]

            self.update[isub] = self.get_update(driver, self.iter)

            if self.update[isub] :
                self.gsystem.density[:] = self.density
                if driver.technique in ['EX', 'MM'] :
                    calcType = ['V']
                else :
                    calcType = ['O']
                driver(gsystem = self.gsystem, calcType = calcType, olevel = self.options['olevel'], sdft = self.options['sdft'])
        self.iter += 1

    def attach(self, function, interval=1, *args, **kwargs):
        self.observers.append((function, interval, args, kwargs))

    def call_observers(self, istep = 0):
        for function, interval, args, kwargs in self.observers:
            call = False
            if interval > 0 and istep % interval == 0:
                call = True
            elif interval <= 0 and istep == abs(interval):
                call = True
            if call: function(*args, **kwargs)

    def optimize(self, **kwargs):
        self.run(**kwargs)

    def run(self, **kwargs):
        for item in self.irun(**kwargs):
            pass
        return item

    def irun(self, **kwargs):
        #-----------------------------------------------------------------------
        sprint('Begin optimize')

        self.guess_pconv()

        self.gsystem.gaussian_density[:] = 0.0
        self.gsystem.density[:] = 0.0
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
            technique = self._get_driver_technique(driver)
            if technique not in ['EX', 'MM'] :
                self.gsystem.update_density(density, isub = i)
                self.gsystem.update_density(gaussian_density, isub = i, fake = True)
                self.gsystem.update_density(core_density, isub = i, core = True)
        sprint('Update density :', self.gsystem.density.integral())
        self.density = self.gsystem.density.copy()
        #-----------------------------------------------------------------------
        self.add_xc_correction()
        #-----------------------------------------------------------------------
        energy_history = [0.0]
        fmt = "{:10s}{:8s}{:24s}{:16s}{:10s}{:10s}{:16s}".format(" ", "Step", "Energy(a.u.)", "dE", "dP", "dC", "Time(s)")
        sprint(fmt)
        seq = "-" * 100
        time_begin = time.time()
        timecost = time.time()
        res_norm = np.ones(self.nsub)
        olevel = self.options['olevel']

        self.converged = False
        yield self.converged
        self.call_observers(self.iter)

        for it in range(self.options['maxiter']):
            #-----------------------------------------------------------------------
            self.step()
            yield self.converged
            self.call_observers(self.iter)
            #-----------------------------------------------------------------------
            self.update_density(update = self.update)
            res_norm = self.get_diff_residual()
            dp_norm = self.get_diff_potential()
            f_str = 'Norm of reidual density : \n'
            f_str += np.array2string(res_norm, separator=' ', max_line_width=80)
            f_str += '\nEnergy of reidual density : \n'
            f_str += np.array2string(dp_norm, separator=' ', max_line_width=80)
            sprint(f_str, level = 2)
            if self.check_converge_potential(dp_norm):
                sprint("#### Subsytem Density Optimization Converged (Potential) In {} Iterations ####".format(it+1))
                self.converged = True
                break
            energy = self.get_energy(self.density, update = self.update, olevel = olevel)[0]
            #-----------------------------------------------------------------------
            energy_history.append(energy)
            timecost = time.time()
            dE = energy_history[-1] - energy_history[-2]
            d_ehart = np.max(dp_norm)
            d_res = np.max(res_norm)
            # d_res= hartree_energy(totalrho-totalrho_prev)
            #-----------------------------------------------------------------------
            fmt = "{:>10s}{:<8d}{:<24.12E}{:<16.6E}{:<10.2E}{:<10.2E}{:<16.6E}".format("Embed: ", self.iter, energy, dE, d_ehart, d_res, timecost - time_begin)
            sprint(seq +'\n' + fmt +'\n' + seq)
            # Only check when accurately calculate the energy
            if olevel == 0 and self.check_converge_energy(energy_history):
                sprint("#### Subsytem Density Optimization Converged (Energy) In {} Iterations ####".format(it+1))
                self.converged = True
                break
            if self.check_stop(): break
        else :
            sprint("!WARN Optimization is exit due to reaching maxium iterations ###")
        self.end_scf()
        yield self.converged

    def set_global_potential(self, **kwargs):
        sdft = self.options['sdft']
        if sdft == 'sdft' :
            self.set_global_potential_sdft(**kwargs)
        elif sdft == 'pdft' :
            self.set_global_potential_pdft(**kwargs)
        self.add_external_potential(**kwargs)

    def set_global_potential_sdft(self, approximate = 'same', **kwargs):
        # approximate = 'density4'

        calcType = ['V']
        if approximate == 'density3' :
            calcType.append('D')
        if self.nsub == len(self.of_drivers):
            pass
        else :
            # initial the global_potential
            self.gsystem.total_evaluator.get_embed_potential(self.gsystem.density, gaussian_density = self.gsystem.gaussian_density, with_global = True, calcType = calcType)
            # static_potential = self.gsystem.total_evaluator.static_potential.copy()
        for isub in range(self.nsub):
            driver = self.drivers[isub]
            if driver is None :
                global_potential = None
                global_density = None
            elif driver.technique == 'OF' :
                continue
            else :
                if driver.evaluator.global_potential is None :
                    if driver.comm.rank == 0 :
                        driver.evaluator.global_potential = Field(grid=driver.grid)
                    else :
                        driver.evaluator.global_potential = driver.atmp
                global_potential = driver.evaluator.global_potential

                if approximate != 'same' :
                    if driver.evaluator.embed_potential is None :
                        if driver.comm.rank == 0 :
                            driver.evaluator.embed_potential = Field(grid=driver.grid)
                        else :
                            driver.evaluator.embed_potential = driver.atmp
                    global_density = driver.evaluator.embed_potential
            self.gsystem.sub_value(self.gsystem.total_evaluator.embed_potential, global_potential, isub = isub)

            if approximate == 'density' or approximate == 'density4' :
                self.gsystem.sub_value(self.gsystem.density, global_density, isub = isub)
                if driver is not None and driver.comm.rank == 0 :
                    factor = np.minimum(np.abs(driver.density/global_density), 1.0)
                    if approximate == 'density4' :
                        alpha = 1E-2
                        factor = factor ** alpha
                    global_potential *= factor
            elif approximate == 'density2' :
                self.gsystem.sub_value(self.gsystem.density, global_density, isub = isub)
                if driver is not None and driver.comm.rank == 0 :
                    factor = np.minimum(np.abs(driver.density/global_density), 1.0)
                    factor = 2.0 - factor
                    global_potential *= factor
            elif approximate == 'density3' :
                self.gsystem.sub_value(self.gsystem.density, global_density, isub = isub)
                if driver is not None :
                    if driver.comm.rank == 0 :
                        energydensity = Field(grid=driver.grid)
                    else :
                        energydensity = driver.atmp
                else :
                    energydensity = None

                self.gsystem.sub_value(self.gsystem.total_evaluator.embed_energydensity, energydensity, isub = isub)

                if driver is not None and driver.comm.rank == 0 :
                    factor = np.maximum((global_density - driver.density)/global_density ** 2, 0.0)
                    factor = np.minimum(factor, 1E6)
                    global_potential += factor * energydensity
        return

    def set_global_potential_pdft(self, approximate = 'same', **kwargs):
        r"""
        .. math::
            v_{PDFT}^{I}= v_{ie}+v_{H}+v_{T_{s}}+v_{XC}-
              \sum_{J \neq I}^{N_{s}}\left(v_{H}^{J}+v_{XC}^{J}+v_{ie}^{J}+v_{T_{s}}^{J}\right) (1)

            v_{PDFT}^{I}= \frac{1}{\rho}\Bigl(\rho \left(v_{ie}+v_{H}+v_{T_{s}}+v_{XC}\right) -
              \sum_{J \neq I}^{N_{s}}\rho^{J}\left(v_{H}^{J}+v_{XC}^{J}+v_{ie}^{J}+v_{T_{s}}^{J}\right)\Bigr) (2)
        """
        approximate = 'density2'
        # approximate = 'density4'

        if approximate == 'density4' :
            total_factor = Field(self.gsystem.grid)
            total_factor[:] = 0.0

        self.gsystem.total_evaluator.get_embed_potential(self.gsystem.density, gaussian_density = self.gsystem.gaussian_density, with_global = True)

        if approximate == 'same' :
            pass
        elif approximate == 'density' :
            self.gsystem.total_evaluator.embed_potential *= self.gsystem.density
        elif approximate == 'density2' or approximate == 'density3' or approximate == 'density4' :
            pass
        else :
            raise AttributeError("{} not supported now".format(approximate))

        for isub in range(self.nsub + len(self.of_drivers)):
            if isub < self.nsub :
                driver = self.drivers[isub]
                if driver is not None and driver.technique == 'OF' : continue
            else :
                driver = self.of_drivers[isub - self.nsub]
                isub = self.of_ids[isub - self.nsub]

            if driver is None :
                global_potential = None
                extpot = None
            else :
                if driver.evaluator.global_potential is None :
                    if driver.technique == 'OF' :
                        driver.evaluator.global_potential = Field(grid=driver.grid)
                    else :
                        if driver.comm.rank == 0 :
                            driver.evaluator.global_potential = Field(grid=driver.grid)
                        else :
                            driver.evaluator.global_potential = driver.atmp

                global_potential = driver.evaluator.global_potential
                global_potential[:] = 0.0

            if approximate != 'density4' :
                if driver is not None :
                    extpot = driver.get_extpot(mapping = False, with_global = False)

            if approximate == 'density' :
                if driver is not None :
                    if driver.technique == 'OF' or driver.comm.rank == 0 :
                        extpot *= driver.density
            elif approximate == 'density2' or approximate == 'density3' :
                self.gsystem.sub_value(self.gsystem.density, global_potential, isub = isub)
                if driver is not None :
                    if driver.technique == 'OF' or driver.comm.rank == 0 :
                        factor = np.minimum(np.abs(driver.density/global_potential), 1.0)
                        extpot *= factor
            elif approximate == 'density4' :
                alpha = 6.0
                self.gsystem.sub_value(self.gsystem.density, global_potential, isub = isub)
                if driver is not None :
                    if driver.technique == 'OF' or driver.comm.rank == 0 :
                        factor = np.minimum((np.abs(driver.density/global_potential)) ** alpha, 1.0)
                    else :
                        factor = None
                    driver.pdft_factor = factor
                else :
                    factor = None

            if approximate == 'density4' :
                self.gsystem.add_to_global(factor, total_factor, isub = isub)
            else :
                self.gsystem.add_to_global(extpot, self.gsystem.total_evaluator.embed_potential, isub = isub)

        if approximate == 'density' :
            self.gsystem.total_evaluator.embed_potential /= self.gsystem.density
        elif approximate == 'density4' :
            mask = total_factor > 1E-3
            total_factor[mask] = 1.0/total_factor[mask]
            for isub in range(self.nsub):
                driver = self.drivers[isub]
                if driver is None :
                    global_potential = None
                    extpot = None
                else :
                    extpot = driver.get_extpot(mapping = False, with_global = False)
                    global_potential = driver.evaluator.global_potential
                    global_potential[:] = 0.0
                self.gsystem.sub_value(total_factor, global_potential, isub = isub)
                if driver is not None :
                    if driver.technique == 'OF' or driver.comm.rank == 0 :
                        factor = global_potential * driver.pdft_factor
                        extpot *= factor

                self.gsystem.add_to_global(extpot, self.gsystem.total_evaluator.embed_potential, isub = isub)

        # write(str(self.iter) + '.xsf', self.gsystem.total_evaluator.embed_potential, self.gsystem.ions)
        # write(str(self.iter) + '_den.xsf', self.gsystem.density, self.gsystem.ions)

        for isub in range(self.nsub):
            driver = self.drivers[isub]
            if driver is None :
                global_potential = None
            else :
                global_potential = driver.evaluator.global_potential
            self.gsystem.sub_value(self.gsystem.total_evaluator.embed_potential, global_potential, isub = isub)
            if approximate == 'density3' :
                if driver is not None :
                    potential = global_potential.copy()
                self.gsystem.sub_value(self.gsystem.density, global_potential, isub = isub)
            if driver is not None :
                if approximate == 'density3' :
                    if driver.technique == 'OF' or driver.comm.rank == 0 :
                        factor = np.minimum(np.abs(driver.density/global_potential), 1.0)
                        driver.evaluator.global_potential = potential * factor
                driver.evaluator.embed_potential = driver.evaluator.global_potential
        return

    def add_external_potential(self, **kwargs):
        for isub in range(self.nsub + len(self.of_drivers)):
            if isub < self.nsub :
                driver = self.drivers[isub]
                if driver is not None and driver.technique == 'OF' : continue
            else :
                driver = self.of_drivers[isub - self.nsub]
                isub = self.of_ids[isub - self.nsub]
            root = 0
            if driver is None :
                global_potential = None
                global_density = None
            else :
                if driver.evaluator.global_potential is None :
                    if driver.technique == 'OF' :
                        driver.evaluator.global_potential = Field(grid=driver.grid)
                    else :
                        if driver.comm.rank == 0 :
                            driver.evaluator.global_potential = Field(grid=driver.grid)
                        else :
                            driver.evaluator.global_potential = driver.atmp
                global_potential = driver.evaluator.global_potential
                if driver.comm.rank == 0 : root = self.gsystem.comm.rank
            root = self.gsystem.grid.mp.amax(root)

            technique = self._get_driver_technique(driver)
            if technique == 'EX' :
                pot = self.gsystem.total_evaluator.external_potential.get('MM', None)
                if pot is not None :
                    # self.gsystem.sub_value(pot, global_potential, isub = isub)
                    pot.gather(out = global_potential, root = root)
                if driver is not None :
                    if driver.density is None :
                        if driver.comm.rank == 0 :
                            driver.density = Field(grid=driver.grid)
                        else :
                            driver.density = driver.atmp
                    global_density = driver.density
                self.gsystem.density.gather(out = global_density, root = root)
            elif technique == 'MM' :
                pot = self.gsystem.total_evaluator.external_potential['QM']
                if pot is None : continue
                pot_en = self.gsystem.total_evaluator.external_potential.get('EX', None)
                if pot_en is not None : pot = pot + pot_en
                # self.gsystem.sub_value(pot, global_potential, isub = isub)
                pot.gather(out = global_potential, root = root)
            else : # if technique in ['OF', 'KS']:
                pot = None
                extpots = self.gsystem.total_evaluator.external_potential
                for key in extpots :
                    if key == 'QM' : continue
                    if pot is None :
                        pot = extpots[key].copy()
                    else :
                        pot += extpots[key]
                if pot is not None :
                    self.gsystem.add_to_sub(pot, global_potential, isub = isub)
        return

    def _get_driver_technique(self, driver):
        techs = {'OF' :0, 'KS' :1, 'EX' :2, 'MM' :3}
        itech = 0
        if driver is not None :
            itech = techs.get(driver.technique, 0)
        itech = self.gsystem.grid.mp.amax(itech)
        for key in techs :
            if itech == techs[key] :
                return key

    def get_update(self, driver, istep):
        update_delay = driver.options.get('update_delay', 0)
        update_freq = driver.options.get('update_freq', 1)
        update_sleep = driver.options.get('update_sleep', 0)
        if istep < update_sleep  :
            update = False
        elif istep > update_delay and (istep - update_delay) % update_freq > 0:
            update = False
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
        self.set_global_potential()
        sdft = self.options['sdft']
        edict = {}
        etype = 1
        if sdft == 'sdft' :
            edict = self.gsystem.total_evaluator(self.gsystem.density, calcType = ['E'], split = True, olevel = 0)
            total_energy = edict.pop('TOTAL').energy
        else :
            if etype == 0 :
                total_energy = np.sum(self.gsystem.total_evaluator.embed_potential * self.gsystem.density) * self.gsystem.grid.dV
                edict['EMB'] = Functional(name = 'ZERO', energy=total_energy + 0, potential=None)
            else :
                edict = self.gsystem.total_evaluator(self.gsystem.density, calcType = ['E'], split = True, olevel = 0)
                total_energy = edict.pop('TOTAL').energy
                # total_func= self.gsystem.total_evaluator(self.gsystem.density, calcType = ['E'], olevel = 0)
                # edict['E_GLOBAL'] = total_func
                # total_energy = total_func.energy.copy()

        etotal, elist = self.get_energy(self.gsystem.density, total_energy, olevel = 0)

        keys = list(edict.keys())
        values = [item.energy for item in edict.values()]
        keys.append('II')
        values.append(self.gsystem.ewald.energy)
        values = self.gsystem.grid.mp.vsum(values)

        # etotal = values.sum()
        # etotal += elist[1:].sum()

        ep_w = OrderedDict()
        for i, key in sorted(enumerate(keys), key=lambda x:x[1]):
            ep_w[key] = values[i]
        for i, item in enumerate(elist[1:]):
            key = "SUB_"+str(i)
            ep_w[key] = item
        key = "TOTAL"
        ep_w[key] = etotal
        sprint(format("Energy information", "-^80"))
        for key, value in ep_w.items():
            sprint("{:>12s} energy: {:22.15E} (eV) = {:22.15E} (a.u.)".format(key, value* ENERGY_CONV["Hartree"]["eV"], value))
        sprint("-" * 80)
        return ep_w

    def end_scf(self):
        for i, driver in enumerate(self.drivers):
            if driver is not None :
                driver.end_scf()
        self.energy_all = self.print_energy()
        self.energy = self.energy_all['TOTAL']
        return

    def stop_run(self, save = ['D'], **kwargs):
        for i, driver in enumerate(self.drivers):
            if driver is not None :
                driver.stop_run(save = save, **kwargs)
        return

    def add_xc_correction(self):
        """
        Sorry, we also hate add the non-linear core correction here, maybe we can find a better way to add it in the future.
        """
        for i, driver in enumerate(self.drivers):
            if driver is None : continue
            if 'XC' in driver.evaluator.funcdicts :
                if hasattr(driver, 'core_density_sub') :
                    driver.evaluator.funcdicts['XC'].core_density = driver.core_density_sub
                else :
                    driver.evaluator.funcdicts['XC'].core_density = driver.core_density
            if driver.technique == 'OF' :
                if 'XC' in driver.evaluator_of.funcdicts :
                    driver.evaluator_of.funcdicts['XC'].core_density = driver.core_density
        if 'XC' in self.gsystem.total_evaluator.funcdicts :
            self.gsystem.total_evaluator.funcdicts['XC'].core_density = self.gsystem.core_density

    def set_kedf_params(self, level = 3, rhotol = 1E-4, thr_ratio = 0.1):
        """
        This is use to set the rhomax for NLKEDF embeddding
        """
        kefunc = self.gsystem.total_evaluator.funcdicts.get('KE', None)
        if kefunc is None : return
        update = False
        if kefunc.name.startswith('STV+GGA+'):
            rhomax = self.gsystem.density.amax() + rhotol
            rho0 = kefunc.rhomax
            if rho0 is None or rho0 < rhomax :
                if level > 2 : update = True
            else :
                rhomax = rho0

            if update :
                sprint('Update the rhomax from {} to {}'.format(rho0, rhomax))
                kefunc.rhomax = rhomax
            kefunc.level = level

            for i, driver in enumerate(self.drivers):
                if driver is None : continue
                if 'KE' in driver.evaluator.funcdicts :
                    if update :
                        driver.evaluator.funcdicts['KE'].rhomax = rhomax
                    driver.evaluator.funcdicts['KE'].level = level

        elif kefunc.name.startswith('MIX_'):
            if level >= 0 : return
            rhomax_all = np.zeros(self.nsub)
            for isub in range(self.nsub):
                driver = self.drivers[isub]
                if driver is None :
                    global_potential = None
                else :
                    if driver.evaluator.global_potential is None :
                        if driver.technique == 'OF' :
                            driver.evaluator.global_potential = Field(grid=driver.grid)
                        else :
                            if driver.comm.rank == 0 :
                                driver.evaluator.global_potential = Field(grid=driver.grid)
                            else :
                                driver.evaluator.global_potential = driver.atmp

                    global_potential = driver.evaluator.global_potential
                    global_potential[:] = 0.0

                self.gsystem.sub_value(self.gsystem.density, global_potential, isub = isub)
                rhomax = 0.0
                if driver is not None :
                    if driver.technique == 'OF' or driver.comm.rank == 0 :
                        mask = np.abs(driver.density/global_potential - 0.5) < thr_ratio
                        ns = np.count_nonzero(mask)
                        if ns > 0 :
                            rho_w = global_potential[mask]
                            rho_s = driver.density[mask]
                            rhomax = rho_w.max() + rho_s.max() + (rho_w - rho_s).max()
                            rhomax *= 3.0/4.0
                    if driver.technique == 'OF' :
                        rhomax = driver.grid.mp.amax(rhomax)
                    else :
                        rhomax = driver.comm.bcast(rhomax, root = 0)

                    if 'KE' in driver.evaluator.funcdicts :
                        driver.evaluator.funcdicts['KE'].rhomax = rhomax

                rhomax = self.gsystem.grid.mp.amax(rhomax)
                rhomax_all[isub] = rhomax

            rhomax = rhomax_all.mean()
            sprint('Update the rhomax :', rhomax_all, rhomax)
            kefunc.rhomax = rhomax
        return

    def output_density(self, suffix = '.snpy', outtype = ['G'], filename = None, **kwargs):
        if 'G' in outtype :
            if filename is None :
                outfile = 'edftpy_gsystem' + suffix
            else :
                outfile = filename
            write(outfile, self.gsystem.density, ions = self.gsystem.ions)

        if 'S' in outtype :
            for i, driver in enumerate(self.drivers):
                if driver is None : continue
                outfile = driver.prefix + suffix
                if driver.technique == 'OF' or driver.comm.rank == 0 or self.gsystem.graphtopo.isub is None:
                    write(outfile, driver.density, ions = driver.subcell.ions)

    def check_stop(self, stopfile = 'edftpy_stopfile', **kwargs):
        stop = False
        if os.path.isfile(stopfile):
            sprint("!WARN Optimization is exit due to the '{}' in {} iterations ###".format(stopfile, self.iter))
            stop = True
        if self.options.get('maxtime', 0) > 1 :
            if self.gsystem.graphtopo.timer.Time('TOTAL') > self.options['maxtime'] :
                sprint("!WARN Optimization is exit due to the maxtime in {} iterations ###".format(self.iter))
                stop = True
        return stop

class MixOptimization(object):
    """
    Mixing TDDFT and SCF

    Notes:
        For the driver of TDDFT, task != 'scf' (restart != 'initial').
    """
    # funcdicts = ['optimize',
    #         'optimizer_tddft', 'optimizer_scf', 'gsystem', 'drivers', 'drivers_fake']

    def __init__(self, optimizer= None, **kwargs):
        self.optimizer_tddft = optimizer
        self.optimizer_scf = optimizer.optimizer
        self.gsystem = self.optimizer_tddft.gsystem

        self.drivers = self.optimizer_tddft.drivers.copy()
        self.drivers_fake = [None for driver in self.drivers]
        for i, driver in enumerate(self.drivers):
            if driver is not None :
                self.drivers_fake[i] = DriverConstraint(driver = driver)
        self.optimizer_tddft.drivers = self.optimizer_tddft.drivers.copy()
        self.optimizer_scf.drivers = self.optimizer_scf.drivers.copy()

    def optimize(self, **kwargs):
        for i, driver in enumerate(self.drivers):
            if driver is not None :
                if driver.task == 'scf' :
                    self.optimizer_tddft.drivers[i] = self.drivers_fake[i]
                else :
                    self.optimizer_scf.drivers[i] = self.drivers_fake[i]

        self.optimizer_tddft.optimize(**kwargs)
        self.optimizer_scf.optimize(**kwargs)

        self.optimizer_tddft.drivers = self.drivers.copy()
        self.optimizer_scf.drivers = self.drivers.copy()

    def __getattr__(self, attr):
        if attr in dir(self):
            return object.__getattribute__(self, attr)
        else :
            return getattr(self.optimizer_scf, attr)
