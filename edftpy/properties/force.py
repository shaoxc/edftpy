import numpy as np

from edftpy.mpi import sprint

def get_total_forces(drivers = None, gsystem = None, linearii=True, shift = True):
    forces = gsystem.get_forces(linearii = linearii)
    # sprint('Total forces0 : \n', forces)
    for i, driver in enumerate(drivers):
        if driver is None : continue
        fs = driver.get_forces()
        ind = driver.subcell.ions_index
        if driver.technique == 'OF' :
            forces[ind] += fs
        elif driver.comm.rank == 0 :
            forces[ind] += fs
        # sprint('ind\n', ind, comm = driver.comm)
        # sprint('fs\n', fs, comm = driver.comm)
    forces = gsystem.grid.mp.vsum(forces)
    #-----------------------------------------------------------------------
    if shift :
        forces_shift = np.mean(forces, axis = 0)
        sprint('Forces shift :', forces_shift)
        forces -= forces_shift
    #-----------------------------------------------------------------------
    sprint('Total forces :')
    sprint(forces)
    return forces

def get_total_stress(drivers = None, gsystem = None, **kwargs):
    stress = gsystem.get_stress()
    for i, driver in enumerate(drivers):
        if driver is None : continue
        fs = driver.get_stress()
        # if driver.comm.rank > 0 : fs = 0.0
        # sprint('fs', fs, flush=True, comm=driver.comm)
        stress += fs
    stress = gsystem.grid.mp.vsum(stress)
    for i in range(2):
        for j in range(i+1, 3):
            stress[j,i] = stress[i,j]
    sprint('Total stress :')
    sprint(stress)
    return stress

def get_total_forces_qmmm(drivers = None, gsystems = None, linearii=True, shift = True):
    '''
    Sep 13, 2023 Xin Chen added. 
    Function for qmmm forces calculation.
    Modified from get_total_force

    '''

    #from dftpy.base import DirectCell, ReciprocalCell, Coord  #should be updated
    #from dftpy.grid import DirectGrid, ReciprocalGrid
    #from dftpy.field import DirectField, ReciprocalField
    #from dftpy.constants import LEN_CONV
    
    def bcast_driver_technique(gsystem, driver):
        techs = {'OF' :0, 'KS' :1, 'EX' :2, 'MM' :3}
        itech = 0
        if driver is not None :
            itech = techs.get(driver.technique, 0)
        itech = gsystem.grid.mp.amax(itech)
        for key in techs :
            if itech == techs[key] :
                return key

    gsystem_qmmm = gsystems[0]
    gsystem_qm   = gsystems[1]
    gsystem_mm   = gsystems[2]
    forces  = gsystem_qmmm.get_forces(linearii = linearii)  # global Ewald+loc !subsystem/subcell.py

    forces_mm = gsystem_mm.get_forces(linearii = linearii)  # MMpart Ewald+loc !subsystem/subcell.py
    indmm = gsystem_mm.ions_index
    #print('MM forces : \n', forces_mm)
    #print('glb forces : \n', forces)
    #forces0 = gsystem_qmmm.grid.mp.vsum(forces)
    #sprint('Total forces0 : \n', forces0)
    forces[indmm] = forces[indmm]-forces_mm
    # forces1 = gsystem_qmmm.grid.mp.vsum(forces)
    #sprint('Total forces1 : \n', forces1)
    #print('after sub forces : \n', forces)
    # Force from drivers.  
    for i, driver in enumerate(drivers):
        if driver is None : continue
        if(driver.technique == 'KS'): fs = driver.get_forces()

        if driver.technique == 'MM' :
            # This block is for MM force calculation
            # Term \Delta(pot)*Z, Hess(pot) \dot dipole.
            continue
            fs = np.reshape(fs, (int(len(fs)/3), 3))
            # Get potential from QM part. XC+H+K

        ind = driver.subcell.ions_index

        if driver.technique == 'OF' :
            forces[ind] += fs
        elif driver.comm.rank == 0 :
            forces[ind] += fs
        #print('Total forces : \n', fs)
    forces = gsystem_qmmm.grid.mp.vsum(forces)



    # Get the potential needed by MM part. 
    embed_keys = ['XC','KE']
    ## MM + QM potential
    #1. XC,KE from [O+H sites + QM part] QM+MM potential
   #gsystem_qmmm.total_evaluator.get_embed_potential(gsystem_qmmm.gaussian_density, embed_keys = embed_keys,
   #        gaussian_density = gsystem_qm.gaussian_density, with_global = False, calcType = ('V'))
    gsystem_qmmm.total_evaluator.get_embed_potential(gsystem_qmmm.gaussian_density, embed_keys = embed_keys,
            gaussian_density = None, with_global = False, calcType = ('V'))
    #2. Hartree+Pseudo from [M+H sites + QM part]
    pot_qmmm = gsystem_qmmm.total_evaluator.get_total_functional(gsystem_qmmm.density, calcType = ('V'), embed_keys = embed_keys).potential
    #3. Total
    #gsystem_qmmm.total_evaluator.embed_potential[:] += pot_qmmm
    #-----------------------------------------------------------------------
    #4. Give potentials above to QM part.
    total_potential = gsystem_qmmm.total_evaluator.embed_potential.copy()
    #total_potential = pot_qmmm
    #-----------------------------------------------------------------------

    ## MM potential
    embed_keys = [ 'XC','KE']
    #1. XC,KE from [O+H sites] 
    gsystem_mm.total_evaluator.get_embed_potential(gsystem_mm.gaussian_density, embed_keys = embed_keys,
            gaussian_density = None, with_global = False, calcType = ('V'))
    #2. Hartree+Pseudo from [M+H sites + QM part]
    pot_mm = gsystem_mm.total_evaluator.get_total_functional(gsystem_mm.density, calcType = ('V'), embed_keys = embed_keys).potential
    #3. Total
    #gsystem_mm.total_evaluator.embed_potential[:] += pot_mm
    #-----------------------------------------------------------------------
    #4. Give potentials above to QM part.
    mm_potential = gsystem_mm.total_evaluator.embed_potential
    #mm_potential = pot_mm
    #-----------------------------------------------------------------------
    
    # QM+MM potential - MM potential
    diff_potential = total_potential - mm_potential

    for i, driver in enumerate(drivers):
        if driver is None : 
            global_potential = None
        else:
            technique = driver.technique
#            print("in loop:", driver.comm.rank, technique)
            driver.evaluator.global_potential = np.zeros_like(driver.density)
            global_potential = driver.evaluator.global_potential
        technique = bcast_driver_technique(gsystem_qmmm,driver)
        
        if technique == 'MM' :
            # Get atomic charges on M, H, H sites for water.
            gsystem_mm.sub_value(diff_potential  , global_potential , isub =i)
            #gsystem_mm.sub_value(gsystem_mm.density, global_potential , isub =i)
            if(not driver is None):
                #print("force dim:",global_potential.shape, technique)
                MM_charges, MM_pos_charges = driver.engine.get_charges() 
                # Get Dipole at O, H, H sites for water 
                MM_dipoles, MM_pos_dipoles = driver.engine.get_dipoles() 
                #print(global_potential.shape)
                #print(MM_pos_charges,MM_charges,driver.engine.get_points_zval())
                pot_grad = global_potential.gradient(flag = 'supersmooth',sigma=0.30)
                pot_hess = global_potential.hessian(flag="supersmooth", sigma=0.3)
                potfield_grad = driver.engine.get_value_at_points(pot_grad, MM_pos_charges).ravel()
                potfield_hess = driver.engine.get_value_at_points(pot_hess, MM_pos_charges).ravel()
                #print("[Force Charge]")
                #print((potfield_grad.reshape((len(MM_charges), 3)).T*(np.array(driver.engine.get_points_zval()) - np.array(MM_charges))).T)
                #print(potfield_hess.reshape((len(MM_charges), 6)), MM_dipoles)

    #-----------------------------------------------------------------------
    # QMMM shift ==None
    shift = False
    if shift :
        forces_shift = np.mean(forces, axis = 0)
        sprint('Forces shift :', forces_shift)
        forces -= forces_shift
    #-----------------------------------------------------------------------
    sprint('Total forces : \n', forces)
    return forces
