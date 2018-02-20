cimport cython
cimport numpy as np
import numpy as np
import ctypes

from ibex.transforms import seg2seg


cdef extern from 'cpp-comparestacks.h':
    void CppEvaluate(long *segmentation, long *gold, long resolution[3], unsigned char mask_ground_truth)

def Evaluate(segmentation, gold, mask_ground_truth=True, filtersize=0):
    # make sure these elements are the same size
    assert (segmentation.shape == gold.shape)

    # remove all small connected components
    if filtersize > 0:
        segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, filtersize)
        gold = seg2seg.RemoveSmallConnectedComponents(gold, filtersize)

    # convert to c++ arrays
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_gold
    cpp_gold = np.ascontiguousarray(gold, dtype=ctypes.c_int64)

    zres, yres, xres = segmentation.shape
    CppEvaluate(&(cpp_segmentation[0,0,0]), &(cpp_gold[0,0,0]), [zres, yres, xres], mask_ground_truth)