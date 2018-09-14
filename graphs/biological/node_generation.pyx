cimport cython
cimport numpy as np
import ctypes
import numpy as np

from ibex.utilities import dataIO
from ibex.data_structures import unionfind;
from ibex.transforms import seg2seg


cdef extern from 'cpp-node_generation.h':
    long *CppFindZSingletons(long *segmentation, long grid_size[3])



def GenerateNodes(prefix, threshold=20000):
    segmentation = dataIO.ReadSegmentationData(prefix)

    import time
    start_time = time.time()

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_grid_size = np.ascontiguousarray(segmentation.shape, dtype=ctypes.c_int64)

    cdef long *matches = CppFindZSingletons(&(cpp_segmentation[0,0,0]), &(cpp_grid_size[0]))

    nmatches = matches[0] / 2

    max_label = np.amax(segmentation)
    union_find = [unionfind.UnionFindElement(iv) for iv in range(max_label)]

    for iv in range(nmatches):
        index_one = matches[2 * iv + 1]
        index_two = matches[2 * iv + 2]

        unionfind.Union(union_find[index_one], union_find[index_two])

    mapping = np.zeros(max_label, dtype=np.int64)
    for iv in range(max_label):
        mapping[iv] = unionfind.Find(union_find[iv]).label

    seg2seg.MapLabels(segmentation, mapping)

    dataIO.WriteH5File(segmentation, 'rhoana/{}-rs.h5'.format(prefix), 'main')


    print nmatches

    print time.time() - start_time