cimport cython
cimport numpy as np
import ctypes
import numpy as np

from ibex.utilities import dataIO
from ibex.utilities.constants import *


cdef extern from 'cpp-teaser_skeletons.h':
    unsigned char *CppGenerateTeaserSkeletons(long *segmentation, long label, long grid_size[3], long world_res[3])

def IndividualSkeleton(segmentation, label, grid_size, world_res, complete_skeletons):
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    
    cdef unsigned char *cpp_skeletons = CppGenerateTeaserSkeletons(&(cpp_segmentation[0,0,0]), label, [grid_size[IB_Z], grid_size[IB_Y], grid_size[IB_X]], [world_res[IB_Z], world_res[IB_Y], world_res[IB_X]])
    cdef unsigned char[:] tmp_skeletons = <unsigned char[:segmentation.size]> cpp_skeletons
    skeletons = np.reshape(np.asarray(tmp_skeletons), grid_size)

    complete_skeletons += label * skeletons

# generate skeletons for this volume
def GenerateTeaserSkeletons(prefix):
    # get the segmentation for this prefix
    segmentation = dataIO.ReadSegmentationData(prefix)

    # get the grid size and world resolution
    grid_size = segmentation.shape
    world_res = dataIO.Resolution(prefix)

    complete_skeletons = np.zeros(grid_size, dtype=np.int64)

    unique = np.unique(segmentation)
    for label in unique:
        IndividualSkeleton(segmentation, label, grid_size, world_res, complete_skeletons)

    dataIO.WriteH5File(complete_skeletons, 'skeletons-mine.h5', 'main')