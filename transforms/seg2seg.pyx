cimport cython
cimport numpy as np
import ctypes
import numpy as np

cdef extern from 'cpp-seg2seg.h':
    unsigned long *CppMapLabels(unsigned long *segmentation, unsigned long *mapping, unsigned long nentries)
    unsigned long *CppRemoveSmallConnectedComponents(unsigned long *segmentation, int threshold, unsigned long nentries)

# map the labels from this segmentation
def MapLabels(segmentation, mapping):
    # get the size of the data
    zres, yres, xres = segmentation.shape
    nentries = segmentation.size

    cdef np.ndarray[unsigned long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_uint64)
    cdef np.ndarray[unsigned long, ndim=1, mode='c'] cpp_mapping
    cpp_mapping = np.ascontiguousarray(mapping, dtype=ctypes.c_uint64)

    cdef unsigned long *mapped_segmentation = CppMapLabels(&(cpp_segmentation[0,0,0]), &(cpp_mapping[0]), nentries)

    cdef unsigned long[:] tmp_segmentation = <unsigned long[:segmentation.size]> mapped_segmentation

    return np.reshape(np.asarray(tmp_segmentation), (zres, yres, xres))

# remove the components less than min size
def RemoveSmallConnectedComponents(segmentation, min_size=64):
    nentries = segmentation.size
    zres, yres, xres = segmentation.shape

    cdef np.ndarray[unsigned long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_uint64)
    
    # call the c++ function
    cdef unsigned long *updated_segmentation = CppRemoveSmallConnectedComponents(&(cpp_segmentation[0,0,0]), min_size, nentries)

    # turn into python numpy array
    cdef unsigned long[:] tmp_segmentation = <unsigned long[:segmentation.size]> updated_segmentation

    # reshape the array to the original shape
    thresholded_segmentation = np.reshape(np.asarray(tmp_segmentation), (zres, yres, xres))	
    return thresholded_segmentation


def ReduceLabels(segmentation):
    # get the unique labels
    unique = np.unique(segmentation)

    # get the maximum label for the segment
    maximum_label = np.amax(segmentation) + 1

    # create an array from original segment id to reduced id
    mapping = np.zeros(maximum_label, dtype=np.int64) - 1

    for ie, label in enumerate(unique):
        mapping[label] = ie

    # return the forward and reverse mapping
    return mapping, unique
