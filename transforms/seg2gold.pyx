cimport cython
cimport numpy as np
import numpy as np
import ctypes

cdef extern from 'cpp-seg2gold.h':
    unsigned long *Seg2Gold(unsigned long *segmentation, unsigned int *gold, long nentries)

def seg2gold(segmentation, gold):
    cdef np.ndarray[unsigned long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_uint64)
    cdef np.ndarray[unsigned int, ndim=3, mode='c'] cpp_gold
    cpp_gold = np.ascontiguousarray(gold, dtype=ctypes.c_uint32)

    max_segmentation = np.amax(segmentation) + 1

    cdef unsigned long *mapping = Seg2Gold(&(cpp_segmentation[0,0,0]), &(cpp_gold[0,0,0]), segmentation.size)
    cdef unsigned long[:] tmp_mapping = <unsigned long[:max_segmentation]> mapping;
    return np.asarray(tmp_mapping)