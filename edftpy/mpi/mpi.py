import numpy as np
from dftpy.mpi import SerialComm
from dftpy.time_data import TimeObj, TimeData

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
        self._sub_index = [None, ] * self.nsub  # the index of subsystem data in region
        self._region_index = [None, ] * self.nsub # the index of global data in region
        self._region_bound = [None, ] * self.nsub # the boundary of region in global
        self.sub_ids = {}

    def sub_index(self, i):
        if self._sub_index[i] is None :
            self._sub_index[i] = self.get_sub_index(i)
        return self._sub_index[i]

    def region_index(self, i):
        if self._region_index[i] is None :
            self._region_index[i] = self.get_region_index(i)
        return self._region_index[i]

    def region_bound(self, i):
        if self._region_bound[i] is None :
            self._region_bound[i] = self.get_region_bound(i)
        return self._region_bound[i]

    def region_shape_sub(self, i):
        b = self.region_bound(i)
        shape = [b[1,0]-b[0,0], b[1,1]-b[0,1], b[1,2]-b[0,2]]
        return np.array(shape)

    def get_sub_index(self, i, in_global = False):
        if in_global :
            indl = self.sub_shift[i]
            ir = self.grid.nrR
        else :
            indl = self.sub_shift[i] - self.region_shift[i]
            ir = self.region_shape[i]
        indr = indl + self.sub_shape[i]
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

    def get_region_bound(self, i):
        indl = self.region_offsets[i]
        indr = indl + self.grid.nr
        bound = np.array([indl, indr])
        return bound


class GraphTopo:
    def __init__(self, comm = None, parallel = False, decomposition = 'Pencil', **kwargs):
        MPI = None
        if comm is None :
            if parallel :
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            else :
                comm = SerialComm(**kwargs)
        #
        self.decomposition = decomposition
        #
        self._is_mpi = parallel
        self._MPI = MPI
        self.isub = None
        self.nprocs = [1]
        self._comm = comm # for global, also used for OF subsystem
        self._comm_sub = None # for subsystem, each processor only belong to one
        self.comm_region = [None] # for the data region of each subsystem
        self._scale_procs = False
        # self.timer = TimeObj()
        self.timer = TimeData
        self.timer.Begin("TOTAL")

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
        self.region_data_buf = [np.zeros(3), ] * self.nsub
        self._region_bound_w = [None, ] * self.nsub # for processor in region
        self._region_subs_shape = [None, ] * self.nsub # for processor in region
        self._region_subs_size = [None, ] * self.nsub # for processor in region
        self._region_subs_index = [None, ] * self.nsub # for processor in region

    def region_bound_w(self, i):
        if self._region_bound_w[i] is None :
            self._region_bound_w[i] = self.get_region_bound_w(i)
            self._region_bound_to_shape(i)
        return self._region_bound_w[i]

    def _region_bound_to_shape(self, i):
        if self.comm_region[i].rank == 0 :
            nrank = self.comm_region[i].size
            region_bound_w = self._region_bound_w[i]
            shape_w = []
            index_w = []
            for ik in range(0, nrank):
                b = region_bound_w[2*ik:2*ik+2]
                ind = np.s_[b[0,0]:b[1,0], b[0,1]:b[1,1], b[0,2]:b[1,2]]
                shape = [b[1,0]-b[0,0], b[1,1]-b[0,1], b[1,2]-b[0,2]]
                shape_w.append(shape)
                index_w.append(ind)
            self._region_subs_index[i] = index_w
            self._region_subs_shape[i] = np.asarray(shape_w, dtype='int')
            self._region_subs_size[i] = self._region_subs_shape[i].prod(axis=1)
        else :
            self._region_subs_index[i] = np.zeros(1)
            self._region_subs_shape[i] = np.zeros(1)
            self._region_subs_size[i] = np.zeros(1)
        return

    def get_region_bound_w(self, i):
        bound = self.graph.region_bound(i)
        if self.comm_region[i].rank == 0 :
            bound_w = np.zeros((2 * self.comm_region[i].size, 3), dtype = 'int')
        else :
            bound_w = np.zeros(3, dtype = 'int')
        if self.comm_region[i].size > 1 :
            self.comm_region[i].Gather(bound, bound_w, root=0)
        else :
            bound_w[:] = bound
        return bound_w

    def region_subs_shape(self, i):
        if self._region_subs_shape[i] is None :
            self.region_bound_w(i)
        return self._region_subs_shape[i]

    def region_subs_size(self, i):
        if self._region_subs_size[i] is None :
            self.region_bound_w(i)
        return self._region_subs_size[i]

    def region_subs_index(self, i):
        if self._region_subs_index[i] is None :
            self.region_bound_w(i)
        return self._region_subs_index[i]

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
                grp_region.Free()
                if ranks[self.rank] < 0 :
                    self.comm_region[i] = None
                else :
                    #reorder the rank of comm_region
                    cmid = i%self.comm_region[i].size
                    ckey = abs(cmid - self.comm_region[i].rank)
                    comm_temp = self.comm_region[i].Split(0, ckey)
                    self.comm_region[i].Free()
                    self.comm_region[i] = comm_temp
                    #
                    self.comm_region[i].Allreduce(offsets, self.graph.region_shift[i], op=self.MPI.MIN)
                    self.graph.region_offsets[i] = offsets - self.graph.region_shift[i]

                    shape = self.graph.region_offsets[i] + grid.nr
                    self.comm_region[i].Allreduce(shape, self.graph.region_shape[i], op=self.MPI.MAX)
            else :
                self.graph.region_shape[i] = nrR
            # print(self.rank, ' region -> ', self.graph.region_shape,  self.graph.region_shift)

    def distribute_procs(self, nprocs = None, scale = False):
        zero = 1E-8 # nonzero
        if not self.is_mpi :
            self.comm_sub = SerialComm()
            self.nprocs = np.ones(1, dtype = 'int')
            return
        elif nprocs is None :
            raise AttributeError("Must give the 'nprocs' in parallel version")
        elif len(nprocs) == 1 and nprocs[0] == 0 :
            self.nprocs = np.ones(1, dtype = 'int')
            return
        nprocs = np.asarray(nprocs, dtype = 'float')
        self.nprocs = np.rint(nprocs).astype(dtype = 'int')
        nmin = np.min(nprocs[nprocs > zero])
        scale = scale or 1.0-nmin > -zero or self._scale_procs
        self._scale_procs = scale
        if self.nprocs.sum() != self.size :
            ns = np.count_nonzero(nprocs > zero)
            if ns == 0 :
                self.comm_sub = SerialComm()
                self.nprocs = np.ones(1, dtype = 'int') * self.size
                return
            if scale : # ratio of processors
                if nprocs.sum() == ns :
                    av = self.size // ns
                    nprocs[:] = nprocs * av
                else :
                    nprocs[:] = nprocs * self.size / nprocs.sum()
                self.nprocs = np.rint(nprocs).astype(dtype = 'int')
            else : # set as maximum of processors
                pass
        if self.nprocs.sum() != self.size :
            res = self.size - self.nprocs.sum()
            while res < 0 :
                self.nprocs[self.nprocs > 1] -= 1
                res = self.size - self.nprocs.sum()
            if scale : # ratio of processors
                for i, n in enumerate(self.nprocs):
                    if res == 0 : break
                    if n > 0 :
                        self.nprocs[i] += 1
                        res -= 1
        ub = 0
        self.isub = len(self.nprocs) + 100
        for i, n in enumerate(self.nprocs):
            ub += n
            if self.rank < ub :
                self.isub = i
                break
        self.comm_sub = self.comm.Split(self.isub, self.rank)
        # print('isub', self.isub, self.rank, self.comm_sub, self.comm)

    def build_region(self, grid = None, drivers = None, **kwargs):
        self.free_comm_region()
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

    def free_comm(self):
        self.free_comm_sub()
        self.free_comm_region()
        return

    def free_comm_sub(self):
        if self.is_mpi :
            if self.comm_sub != self.comm :
                self.comm_sub.Free()
                self.comm_sub = self.comm
        return

    def free_comm_region(self):
        for i, comm in enumerate(self.comm_region):
            if comm is not None :
                comm.Free()
                self.comm_region[i] = None
            # self.comm.Barrier()
        return

    @property
    def is_mpi(self):
        # if isinstance(self._comm, SerialComm):
        if self._comm.size < 2 :
            self._is_mpi = False
        else :
            self._is_mpi = True
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
    def comm_sub(self):
        if self._comm_sub is None :
            return self.comm
        else :
            return self._comm_sub

    @comm_sub.setter
    def comm_sub(self, value):
        self._comm_sub = value

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

    def region_to_sub(self, i, total, sub_data):
        self.creat_tmp_data(i)
        index = self.graph.sub_index(i)
        if self.is_mpi :
            # gather the data to all region processors
            #-----------------------------------------------------------------------
            if self.comm_region[i] is not None :
                if self.comm_region[i].size > 1 :
                    count = self.region_subs_size(i)
                    displ = np.cumsum(count) - count
                    self.comm_region[i].Gatherv(total, [self.region_data_buf[i], count, displ, self.MPI.DOUBLE], root=0)

                    if self.rank == self.rank_region[i] :
                        nrank = self.comm_region[i].size
                        shape_w = self.region_subs_shape(i)
                        index_w = self.region_subs_index(i)
                        for ik in range(0, nrank):
                            ind = index_w[ik]
                            shape = shape_w[ik]
                            self.region_data[i][ind] = self.region_data_buf[i][displ[ik]:displ[ik]+count[ik]].reshape(shape)
                else : # only one processor in region
                    self.region_data[i] = total
            #-----------------------------------------------------------------------

            if self.rank_region[i] == self.rank_sub[i] == self.rank:
                sub_data[:] = self.region_data[i][index]
            else :
                if self.rank == self.rank_region[i] :
                    buf = np.empty(self.graph.sub_shape[i])
                    buf[:] = self.region_data[i][index]
                    self.comm.Send(buf, dest = self.rank_sub[i], tag = i)
                elif self.rank == self.rank_sub[i] :
                    self.comm.Recv(sub_data, source=self.rank_region[i], tag = i)
            self.comm.Barrier()
        else :
            sub_data[:] = self.region_data[i][index]

    def sub_to_region(self, i, sub_data):
        self.creat_tmp_data(i)
        recv_buf = None
        index = self.graph.sub_index(i)
        if self.is_mpi :
            if self.rank_region[i] == self.rank_sub[i] == self.rank :
                self.region_data[i][index] = sub_data
            else :
                if self.rank == self.rank_sub[i] :
                    self.comm.Send(sub_data, dest = self.rank_region[i], tag = i)
                elif self.rank == self.rank_region[i] :
                    buf = np.empty(self.graph.sub_shape[i])
                    self.comm.Recv(buf, source = self.rank_sub[i], tag = i)
                    self.region_data[i][index] = buf
            self.comm.Barrier()
            #-----------------------------------------------------------------------
            if self.comm_region[i] is not None :

                count = self.region_subs_size(i)
                displ = np.cumsum(count) - count

                if self.comm_region[i].size > 1 :
                    if self.rank == self.rank_region[i] :
                        nrank = self.comm_region[i].size
                        shape_w = self.region_subs_shape(i)
                        index_w = self.region_subs_index(i)
                        for ik in range(0, nrank):
                            ind = index_w[ik]
                            shape = shape_w[ik]
                            # arr = self.region_data[i][ind]
                            self.region_data_buf[i][displ[ik]:displ[ik]+count[ik]] = self.region_data[i][ind].ravel()
                    shape = self.graph.region_shape_sub(i)
                    recv_buf = np.empty(shape)
                    self.comm_region[i].Scatterv([self.region_data_buf[i], count, displ, self.MPI.DOUBLE], recv_buf, root=0)
                else :
                    ind = self.graph.region_index(i)
                    return self.region_data[i][ind]
            #-----------------------------------------------------------------------
        else :
            self.region_data[i][index] = sub_data
            recv_buf = self.region_data

        return recv_buf

    def creat_tmp_data(self, i):
        if self.comm_region[i] is not None and self.rank == self.rank_region[i] :
            if np.all(self.graph.region_shape[i] == np.array(self.region_data[i].shape)):
                self.region_data[i][:] = 0.0
                self.region_data_buf[i][:] = 0.0
            else :
                self.region_data[i] = np.zeros(self.graph.region_shape[i])
                self.region_data_buf[i] = np.zeros(self.graph.region_shape[i].prod())

    def _sub_to_global_serial(self, i, sub_data, total = None, overwrite = False, index = None):
        if index is None :
            index = self.graph.sub_index(i)
        if overwrite :
            total[index] = sub_data
        else :
            total[index] += sub_data

    def sub_to_global(self, sub_data, total, isub = None, grid = None, overwrite = False):
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
        else :
            value = self.sub_to_region(i, sub_data)
            if self.comm_region[i] is not None :
                if overwrite :
                    total[:] = value
                else :
                    total[:] += value

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
            self.region_to_sub(i, total, sub_data)
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
