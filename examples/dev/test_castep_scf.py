import numpy as np
import caspytep

from ase.lattice.cubic import Diamond, FaceCenteredCubic
from ase.io.castep import write_cell, write_param

seed = 'castep_in'

caspytep.cell.cell_read(seed)

current_cell = caspytep.cell.get_current_cell()
caspytep.ion.ion_read()

real_charge = float(np.dot(current_cell.num_ions_in_species, current_cell.ionic_charge))
real_num_atoms = current_cell.mixture_weight.sum()
fixed_cell = current_cell.cell_constraints.max() == 0

caspytep.parameters.parameters_read(seed, real_charge, real_num_atoms, fixed_cell)
current_params = caspytep.parameters.get_current_params()

caspytep.comms.comms_parallel_strategy(current_params.data_distribution,
                                     current_params.nbands,
                                     current_cell.nkpts,
                                     current_params.num_farms_requested,
                                     current_params.num_proc_in_smp)

caspytep.cell.cell_distribute_kpoints()
caspytep.ion.ion_initialise()

caspytep.parameters.parameters_output(caspytep.stdout)
caspytep.cell.cell_output(caspytep.stdout)

caspytep.basis.basis_initialise(current_params.cut_off_energy)
current_basis = caspytep.basis.get_current_basis()
print('Number of points :', current_basis.num_real_points,current_basis.num_fine_real_points)
current_params.fine_gmax = (current_params.fine_grid_scale *
                            np.sqrt(2.0*current_params.cut_off_energy))
caspytep.ion.ion_real_initialise()

main_model = caspytep.model.model_state()
main_model.converged = caspytep.electronic.electronic_minimisation(main_model)
if not main_model.converged:
    raise RuntimeError('Electronic minimisation did not converge!')

print('cutoff', current_params.cut_off_energy, 'energy', main_model.total_energy)
