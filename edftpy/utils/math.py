import numpy as np
import scipy.special as sp
from scipy import ndimage
from scipy import signal

def gaussian(x, sigma = 0.4, mu = 0.0, dim = 3):
    if dim > 1 :
        y = 1.0/(np.sqrt(2.0 * np.pi) * sigma) ** dim * np.exp(x * x /(-2.0 * sigma * sigma))
    else :
        y = 1.0/(np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu)/sigma) ** 2.0)
    return y
