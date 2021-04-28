from edftpy.mixer.mixer import AbstractMixer, SpecialPrecondition


class LinearMixer(AbstractMixer):
    def __init__(self, predtype = None, predcoef = [0.8, 1.0], coef = 0.5, delay = 3, predecut = None, **kwargs):
        # if predtype is None :
            # self.pred = None
        self.pred = SpecialPrecondition(predtype, predcoef, predecut=predecut)
        self.coef = coef
        self._delay = delay
        self.restart()

    def restart(self):
        self._iter = 0

    def __call__(self, nin, nout, coef = None):
        results = self.compute(nin, nout, coef)
        return results

    def compute(self, nin, nout, coef = None):
        self._iter += 1
        one = 1.0-1E-10
        if coef is None :
            coef = self.coef
        if self._iter > self._delay and coef < one:
            res = nout - nin
            res *= coef
            results = self.pred(nin, nout, residual=res, coef=coef)
            results = self.format_density(results, nin)
        else :
            results = nout.copy()

        return results

    def residual(self, nin, nout):
        res = nout - nin
        return res
