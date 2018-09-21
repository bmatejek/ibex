cimport cython
cimport numpy as np
import ctypes
from libcpp cimport bool
import numpy as np
import scipy.ndimage
import time

from ibex.utilities import dataIO


cdef extern from 'cpp-seg2seg.h':
    void CppMapLabels(long *segmentation, long *mapping, unsigned long nentries)
    void CppRemoveSmallConnectedComponents(long *segmentation, int threshold, unsigned long nentries)
    void CppForceConnectivity(long *segmentation, long grid_size[3])
    void CppDownsampleMapping(const char *prefix, long *segmentation, long input_resolution[3], long output_resolution[3], long input_grid_size[3], bool benchmark)
    


# map the labels from this segmentation
def MapLabels(segmentation, mapping):
    # convert segmentation to int64
    if not segmentation.dtype == np.int64: segmentation = segmentation.astype(np.int64)
    mapping = np.copy(mapping).astype(np.int64)

    # get the size of the data
    nentries = segmentation.size

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_mapping = np.ascontiguousarray(mapping, dtype=ctypes.c_int64)

    CppMapLabels(&(cpp_segmentation[0,0,0]), &(cpp_mapping[0]), nentries)



# remove the components less than min size
def RemoveSmallConnectedComponents(segmentation, threshold=64):  
    if threshold == 0: return segmentation

    # convert segmentation to int64
    if not segmentation.dtype == np.int64: segmentation = segmentation.astype(np.int64)

    nentries = segmentation.size
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation= np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    
    # call the c++ function
    CppRemoveSmallConnectedComponents(&(cpp_segmentation[0,0,0]), threshold, nentries)



# reduce the labeling
def ReduceLabels(segmentation):
    # get the unique labels
    unique = np.unique(segmentation)

    # get the maximum label for the segment
    maximum_label = np.amax(segmentation) + 1

    # create an array from original segment id to reduced id
    mapping = np.zeros(maximum_label, dtype=np.int64) - 1
    # extraceullar maps to extracellular
    mapping[0] = 0

    # nothing else should map to zero
    index = 1
    for label in unique:
        # prevent extracellular material
        if label == 0: continue

        # set the mapping to this index and increment
        mapping[label] = index
        index += 1

    # return the forward and reverse mapping
    return mapping, unique



def ForceConnectivity(segmentation):
    # convert segmentation to int64
    if not segmentation.dtype == np.int64: segmentation = segmentation.astype(np.int64)

    # transform into c array
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_grid_size = np.ascontiguousarray(segmentation.shape, dtype=ctypes.c_int64)

    # call the c++ function
    CppForceConnectivity(&(cpp_segmentation[0,0,0]), &(cpp_grid_size[0]))
    
    del cpp_grid_size



def DownsampleMapping(prefix, output_resolution=(80, 80, 80), benchmark=False):
    # benchmark data uses gold
    if benchmark: segmentation = dataIO.ReadGoldData(prefix)
    else: segmentation = dataIO.ReadSegmentationData(prefix)

    # convert segmentation to int64
    if not segmentation.dtype == np.int64: segmentation = segmentation.astype(np.int64)

    # ignore time to read data
    start_time = time.time()

    input_resolution = dataIO.Resolution(prefix)

    # convert numpy arrays to c++ format
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_input_resolution = np.ascontiguousarray(input_resolution, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_output_resolution = np.ascontiguousarray(output_resolution, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_input_grid_size = np.ascontiguousarray(segmentation.shape, dtype=ctypes.c_int64)

    # call c++ function
    CppDownsampleMapping(prefix, &(cpp_segmentation[0,0,0]), &(cpp_input_resolution[0]), &(cpp_output_resolution[0]), &(cpp_input_grid_size[0]), benchmark)

    # free memory
    del cpp_segmentation
    del cpp_input_resolution
    del cpp_output_resolution
    del cpp_input_grid_size

    print 'Downsampling to resolution {} in {} seconds'.format(output_resolution, time.time() - start_time)