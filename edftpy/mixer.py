import numpy as np
import scipy.special as sp

from abc import ABC, abstractmethod
import warnings


class AbstractMixer(ABC):
    """
    This is a template class for mixer
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class LinearMixer(AbstractMixer):
    def __init__(self, predcond = None, predcoef = [0.8, 0.1]):
        self.predcond = predcond
        self.predcoef = predcoef

    def __call__(self, prev, now, coef = [0.5]):
        if self.predcond is None :
            a1 = coef[0]
            a2 = 1.0 - coef[0]
            results = prev * a1 + now * a2
        elif self.predcond == 'kerker' :
            A = self.predcoef[0]
            q0 = self.predcoef[1] ** 2
            rhoinG = prev.fft()
            rhooutG = now.fft()
            gg = prev.grid.get_reciprocal().gg
            results = rhoinG + A * gg/(gg+q0)*rhooutG
            results = results.ifft(force_real=True)
        return results
