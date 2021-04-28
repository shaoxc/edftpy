from dftpy.pseudo import LocalPseudo, ReadPseudo


class LocalPP(LocalPseudo):
    def __init__(self, grid=None, ions=None, PP_list=None, PME=True, **kwargs):
        # obj = super().__init__(grid = grid, ions = ions, PP_list = PP_list, PME = PME, MaxPoints = 150000, **kwargs)
        obj = super().__init__(grid = grid, ions = ions, PP_list = PP_list, PME = PME, **kwargs)
        return obj
