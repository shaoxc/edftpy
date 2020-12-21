import numpy as np
from dftpy.mpi import SerialComm

class Graph :
    def __init__(self, nsub = 1, grid = None, **kwargs):
        self.nsub = nsub
        self._default_vars()
        self.grid = grid

    def _default_vars(self):
        self.sub_shift = np.zeros((self.nsub, 3), dtype = 'int')
        self.sub_shape = np.zeros((self.nsub, 3), dtype = 'int')
        self.region_offsets = np.zeros((self.nsub, 3), dtype = 'int')
        self.region_shape = np.zeros((self.nsub, 3), dtype = 'int')
        self.region_shift = np.zeros((self.nsub, 3), dtype = 'int')
        self._sub_index = [None, ] * self.nsub
        self._region_index = [None, ] * self.nsub # for processor in region
        self.sub_ids = {}

    def sub_index(self, i):
        if self._sub_index[i] is None :
            self._sub_index[i] = self.get_sub_index(i)
        return self._sub_index[i]

    def region_index(self, i):
        if self._region_index[i] is None :
            self._region_index[i] = self.get_region_index(i)
        return self._region_index[i]

    def get_sub_index(self, i):
        indl = self.sub_shift[i] - self.region_shift[i]
        indr = indl + self.sub_shape[i]
        ir = self.region_shape[i]
        # print('get_sub_index', indl, indr, ir)
        if np.all(indl > -1) and np.all(indr < ir + 1) :
            index = np.s_[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]]
        else :
            index = []
            for i in range(3):
                if indl[i] < 0 :
                    im = indl[i] + ir[i]
                    idx = list(range(im, ir[i]))
                    idx.extend(list(range(indr[i])))
                elif indr[i] > ir[i] :
                    im = indr[i] - ir[i]
                    idx = list(range(indl[i], ir[i]))
                    idx.extend(list(range(im)))
                else :
                    idx = list(range(indl[i], indr[i]))
                index.append(idx)
            index = np.ix_(*index)
        return index

    def get_region_index(self, i):
        indl = self.region_offsets[i]
        indr = indl + self.grid.nr
        index = np.s_[indl[0]:indr[0], indl[1]:indr[1], indl[2]:indr[2]]
        return index


class GraphTopo:
    def __init__(self, comm = None, parallel = False, **kwargs):
        MPI = None
        if comm is None :
            if parallel :
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            else :
                comm = SerialComm(**kwargs)
        self._comm = comm
        self._is_mpi = parallel
        self._MPI = MPI
        self.isub = None
        self.comm_sub = None

    def _set_default_vars(self, grid = None, drivers = None):
        self.grid = grid
        self.drivers = drivers
        self.nsub = len(self.drivers)
        self.graph = Graph(self.nsub, grid)
        # save the rank of regions and sub
        self.rank_region = np.zeros(self.nsub, dtype='int')
        self.rank_sub = np.zeros(self.nsub, dtype='int')
        self.comm_region= [None, ] * self.nsub
        self.region_data = [np.zeros(3), ] * self.nsub

    def grid_to_region(self, grid=None):
        ranks = np.empty(self.comm.size, dtype = 'int')
        indl_local = grid.offsets
        indr_local = indl_local + grid.nr
        nrR = grid.nrR
        rpbc = np.mgrid[-1:2,-1:2,-1:2].reshape((3,-1)).T

        for i in range(self.nsub):
            ranks[:] = -1
            indl = self.graph.sub_shift[i]
            indr = indl + self.graph.sub_shape[i]
            # check the local data in the subsystem
            #-----------------------------------------------------------------------
            lflag = False
            for r in rpbc :
                il = indl_local + r * nrR
                ir = indr_local + r * nrR
                for k in range(3):
                    # print(r, il[k], ir[k], self.rank)
                    if ir[k] < indl[k] + 1 : break
                    if il[k] > indr[k] - 1 : break
                else :
                    lflag = True
                    break
            #-----------------------------------------------------------------------
            # print('lflag', lflag, indl_local, indr_local, indl,indr, il, ir, self.rank)
            if lflag :
                offsets = il
                ranks[self.rank] = self.rank
            else :
                ranks[self.rank] = -1
            if self.is_mpi :
                self.comm.Allgather(self.MPI.IN_PLACE, ranks)
                grp_region= self.group.Incl(ranks[ranks > -1])
                self.comm_region[i] = self.comm.Create(grp_region)
                if ranks[self.rank] < 0 :
                    self.comm_region[i] = None
                else :
                    self.comm_region[i].Allreduce(offsets, self.graph.region_shift[i], op=self.MPI.MIN)
                    self.graph.region_offsets[i] = offsets - self.graph.region_shift[i]

                    shape = self.graph.region_offsets[i] + grid.nr
                    self.comm_region[i].Allreduce(shape, self.graph.region_shape[i], op=self.MPI.MAX)
            else :
                self.graph.region_shape[i] = nrR
            # print(self.rank, ' region -> ', self.graph.region_shape,  self.graph.region_shift)

    def distribute_procs(self, nprocs = None):
        if not self.is_mpi :
            self.comm_sub = SerialComm()
            return
        if sum(nprocs) != self.comm.size :
            ns = np.count_nonzero(np.asarray(nprocs) > 0)
            if ns == 0 :
                self.comm_sub = SerialComm()
                return
            av = self.size // ns
            res = self.size - av * ns
            for i, n in enumerate(nprocs):
                if n > 0 :
                    if res > 0 :
                        nprocs[i] = av + 1
                        res -= 1
                    else :
                        nprocs[i] = av
        ub = 0
        for i, n in enumerate(nprocs):
            ub += n
            if self.rank < ub :
                self.isub = i
                break
        self.comm_sub = self.comm.Split(self.isub, self.rank)
        # print('isub', self.isub, self.rank, self.comm_sub, self.comm)

    def build_region(self, grid = None, drivers = None, **kwargs):
        self._set_default_vars(grid = grid, drivers = drivers, **kwargs)
        of_id = {}
        for i, driver in enumerate(self.drivers):
            if driver is not None :
                if driver.technique == 'OF' :
                    of_id[id(driver.grid)] = i
                    if self.rank == 0 :
                        self.graph.sub_shape[i] = driver.subcell.grid.nrR
                        self.graph.sub_shift[i] = driver.subcell.grid.shift
                else :
                    # self.graph.sub_ids[id(driver.subcell.grid)] = i
                    self.graph.sub_ids[id(driver.grid)] = i
                    if self.comm_sub.rank == 0 :
                        self.rank_sub[i] = self.rank
                        self.graph.sub_shape[i] = driver.subcell.grid.nrR
                        self.graph.sub_shift[i] = driver.subcell.grid.shift

        if self.is_mpi :
            self.comm.Allreduce(self.MPI.IN_PLACE, self.rank_sub, op=self.MPI.SUM)
            self.comm.Allreduce(self.MPI.IN_PLACE, self.graph.sub_shape, op=self.MPI.SUM)
            self.comm.Allreduce(self.MPI.IN_PLACE, self.graph.sub_shift, op=self.MPI.SUM)
        self.graph.sub_ids.update(of_id)
        # print(self.rank, ' sub -> ', self.graph.sub_shape,self.graph.sub_shift, driver.subcell.grid.nrR)

        self.grid_to_region(self.grid)

        if self.is_mpi :
            for i, driver in enumerate(self.drivers):
                if self.comm_region[i] is not None and self.comm_region[i].rank == 0 :
                    self.rank_region[i] = self.rank
            self.comm.Allreduce(self.MPI.IN_PLACE, self.rank_region, op=self.MPI.SUM)

    @property
    def is_mpi(self):
        self._is_mpi = True
        if isinstance(self._comm, SerialComm):
        # if self._comm.size < 2 :
            self._is_mpi = False
        return self._is_mpi

    @property
    def rank(self):
        return self.comm.rank

    @property
    def size(self):
        return self.comm.size

    @property
    def group(self):
        return self.comm.group

    @property
    def comm(self):
        return self._comm

    @comm.setter
    def comm(self, value):
        self._comm = value

    @property
    def is_root(self):
        self._is_root = self.comm.rank == 0
        return self._is_root

    @property
    def MPI(self):
        if self.is_mpi :
            if self._MPI is None :
                from mpi4py import MPI
                self._MPI = MPI
            return self._MPI
        else :
            raise AttributeError("Only works for parallel version")

    def region_to_sub(self, i, sub_data):
        index = self.graph.sub_index(i)
        # print('region_to_sub', np.sum(sub_data))
        if self.is_mpi :
            # gather the data to all region processors
            if self.comm_region[i] is not None and self.comm_region[i].size > 1 :
                if self.rank == self.rank_region[i] :
                    self.comm_region[i].Reduce(self.MPI.IN_PLACE, self.region_data[i], op=self.MPI.SUM, root=0)
                else :
                    self.comm_region[i].Reduce(self.region_data[i], None, op=self.MPI.SUM, root=0)

            if self.rank_region[i] == self.rank_sub[i] == self.rank:
                sub_data[:] = self.region_data[i][index]
            else :
                if self.rank == self.rank_region[i] :
                    self.comm.Isend(self.region_data[i][index], dest = self.rank_sub[i], tag = i)
                elif self.rank == self.rank_sub[i] :
                    recv_req = self.comm.Irecv(sub_data, source= self.rank_region[i], tag = i)
                    recv_req.wait()

            if self.isub == i :
                self.comm_sub.Bcast(sub_data, root=0)
        else :
            sub_data[:] = self.region_data[i][index]

    def sub_to_region(self, i, sub_data):
        index = self.graph.sub_index(i)
        if self.is_mpi :
            if self.rank_region[i] == self.rank_sub[i] == self.rank :
                self.region_data[i][index] = sub_data
            else :
                if self.rank == self.rank_sub[i] :
                    self.comm.Isend(sub_data, dest = self.rank_region[i], tag = i)
                elif self.rank == self.rank_region[i] :
                    recv_req = self.comm.Irecv(self.region_data[i][index], source = self.rank_sub[i], tag = i)
                    recv_req.wait()
        else :
            self.region_data[i][index] = sub_data

        # bcast the data to all region processors
        if self.comm_region[i] is not None and self.comm_region[i].size > 1 :
            self.comm_region[i].Bcast(self.region_data[i], root=0)

    def creat_tmp_data(self, i):
        if self.comm_region[i] is not None :
            if np.all(self.graph.region_shape[i] == np.array(self.region_data[i].shape)):
                self.region_data[i][:] = 0.0
            else :
                self.region_data[i] = np.zeros(self.graph.region_shape[i])

    def _sub_to_global_serial(self, i, sub_data, total = None, overwrite = False, index = None):
        if index is None :
            index = self.graph.sub_index(i)
        if overwrite :
            total[index] = sub_data
        else :
            total[index] += sub_data

    def sub_to_global(self, sub_data, total, isub = None, grid = None, overwrite = False):
        #-----------------------------------------------------------------------
        lflag = 1
        if sub_data is None :
            lflag = 0
        if self.is_mpi :
            lflag = self.comm.allreduce(lflag, op=self.MPI.MAX)
        if not lflag : return
        #-----------------------------------------------------------------------
        i = self.data_to_isub(sub_data, isub = isub, grid = grid)
        if not self.is_mpi :
            self._sub_to_global_serial(i, sub_data, total, overwrite = overwrite)
            return
        elif self.drivers[i] is not None and self.drivers[i].technique == 'OF' :
            index = slice(None)
            self._sub_to_global_serial(i, sub_data, total, overwrite = overwrite, index = index)
            return
        self.creat_tmp_data(i)
        self.sub_to_region(i, sub_data)
        if self.comm_region[i] is not None :
            index = self.graph.region_index(i)
            if overwrite :
                total[:] = self.region_data[i][index]
            else :
                total[:] += self.region_data[i][index]

    def _global_to_sub_serial(self, i, total, sub_data = None, add = False, index = None):
        if index is None :
            index = self.graph.sub_index(i)
        if sub_data is None :
            sub_data = total[index]
        elif add :
            sub_data[:] += total[index]
        else :
            sub_data[:] = total[index]
        return sub_data

    def global_to_sub(self, total, sub_data = None, isub = None, grid = None, add = False):
        i = self.data_to_isub(sub_data, isub = isub, grid = grid)
        if not self.is_mpi :
            return self._global_to_sub_serial(i, total, sub_data, add=add)
        elif self.drivers[i] is not None and self.drivers[i].technique == 'OF' :
            index = slice(None)
            return self._global_to_sub_serial(i, total, sub_data, add=add, index=index)
        else :
            if add and self.isub == i :
                saved = sub_data.copy()
            if self.comm_region[i] is not None :
                self.creat_tmp_data(i)
                index = self.graph.region_index(i)
                # print('index', index, i, self.region_data[i].shape)
                self.region_data[i][index] = total
            self.region_to_sub(i, sub_data)
            # print('sub_data', sub_data.shape)
            if add and self.isub == i :
                sub_data[:] += saved
        return sub_data

    def data_to_isub(self, sub_data = None, isub = None, grid = None):
        if isub is None :
            if grid is None and sub_data is None :
                isub = -1
            else :
                if grid is None :
                    grid = sub_data.grid
                isub = self.graph.sub_ids[id(grid)]
            if self.is_mpi :
                isub=self.comm.allreduce(isub, op=self.MPI.MAX)
        return isub