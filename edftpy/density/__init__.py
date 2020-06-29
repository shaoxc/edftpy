from edftpy.utils.common import Field
import numpy as np

def normalization_density(density, ncharge = None, grid = None, tol = 1E-300):
    #-----------------------------------------------------------------------
    # minn = np.min(density) - tol
    # density -= minn
    #-----------------------------------------------------------------------
    print('min0', np.min(density))
    print('total0', np.sum(density)*grid.dV)
    # density[density < tol] = tol
    if ncharge is not None :
        if grid is None :
            grid = density.grid
        density += (ncharge-np.sum(density)*grid.dV)/np.size(density)
        density = Field(grid, data=density, direct=True)
    print('min1', np.min(density))
    print('total1', np.sum(density)*grid.dV)
    return density
