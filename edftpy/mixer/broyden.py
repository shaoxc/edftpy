import numpy as np

from edftpy.mixer.mixer import AbstractMixer, SpecialPrecondition


# class BroydenMixer(AbstractMixer):
#     """
#     Not finished !!!
#     """
#     def __init__(self, predtype = None, predcoef = [0.8, 0.1], maxm = 5, coef = 1.0, delay = 3):
#         self.pred = SpecialPrecondition(predtype, predcoef)
#         self._delay = delay
#         self.maxm = maxm
#         self.coef = coef
#         self.restart()

#     def __call__(self, nin, nout, coef = None):
#         results = self.compute(nin, nout, coef)
#         return results

#     def restart(self):
#         self._iter = 0
#         self.dr_mat = None
#         self.dn_mat = None
#         self.jr_mat = None
#         self.prev_density = None
#         self.prev_residual = None

#     def residual(self, nin, nout):
#         res = nout - nin
#         return res

#     def compute(self, nin, nout, coef = None):
#         """
#         Ref : G. Kresse and J. Furthmuller, Comput. Mat. Sci. 6, 15-50 (1996).
#         """
#         self.comm = nin.grid.mp.comm
#         if coef is None :
#             coef = self.coef
#         self._iter += 1

#         r = nout - nin
#         if self._iter > self._delay :
#             dn = nin - self.prev_density
#             dr = r - self.prev_residual
#             if self.dr_mat is None :
#                 self.dr_mat = dr.reshape((-1, *dr.shape))
#                 self.dn_mat = dn.reshape((-1, *dn.shape))
#                 self.jr_mat = np.empty_like(self.dr_mat)
#             elif len(self.dr_mat) < self.maxm :
#                 self.dr_mat = np.concatenate((self.dr_mat, dr.reshape((-1, *r.shape))))
#                 self.dn_mat = np.concatenate((self.dn_mat, dn.reshape((-1, *r.shape))))
#                 self.jr_mat = np.concatenate((self.jr_mat, np.empty((1, *r.shape))))
#             else :
#                 self.dr_mat = np.roll(self.dr_mat,-1,axis=0)
#                 self.dr_mat[-1] = dr
#                 self.dn_mat = np.roll(self.dn_mat,-1,axis=0)
#                 self.dn_mat[-1] = dn
#                 self.jr_mat = np.roll(self.jr_mat,-1,axis=0)

#             ns = len(self.dr_mat)
#             amat = np.empty((ns, ns))
#             x = np.empty((ns))
#             for i in range(ns):
#                 for j in range(i + 1):
#                     amat[i, j] = np.sum(self.dr_mat[i] * self.dr_mat[j])
#                     amat[j, i] = amat[i, j]
#             self.jr_mat[-1] = self.dn_mat[-1].copy()
#             self.jr_mat[-1] = self.pred(self.jr_mat[-1], self.dr_mat[-1]*coef, grid=nin.grid)
#             for i in range(ns - 1):
#                 self.jr_mat[-1] -= amat[i, -1] * self.jr_mat[-1]
#             self.jr_mat[-1] /= amat[-1, -1]
#             for i in range(ns):
#                 x[i] = -np.sum(self.dr_mat[i] * r)

#             for i in range(ns):
#                 if i == 0 :
#                     results = nin + x[i] * self.jr_mat[i]
#                 else :
#                     results += x[i] * self.jr_mat[i]
#             results = self.pred(results, r*coef, nin.grid)
#         else :
#             results = nout.copy()
#         #-----------------------------------------------------------------------
#         results = self.format_density(results, nin)
#         #-----------------------------------------------------------------------
#         self.prev_density = nin.copy()
#         self.prev_residual = r.copy()

#         return results
