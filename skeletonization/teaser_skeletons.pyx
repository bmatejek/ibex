cimport cython
cimport numpy as np
import ctypes
import numpy as np
import time

from ibex.skeletonization.util import PreprocessSegment
from ibex.utilities import dataIO
from ibex.utilities.constants import *


cdef extern from 'cpp-teaser_skeletons.h':
    unsigned char *CppGenerateTeaserSkeletons(long *segmentation, long grid_size[3], long world_res[3])



def IndividualSkeleton(segmentation, world_res):
    # the grid size changes per segment
    grid_size = segmentation.shape

    # convert the array to c++ style
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    
    # generate the skeleton for this element
    cdef unsigned char *cpp_skeletons = CppGenerateTeaserSkeletons(&(cpp_segmentation[0,0,0]), [grid_size[IB_Z], grid_size[IB_Y], grid_size[IB_X]], [world_res[IB_Z], world_res[IB_Y], world_res[IB_X]])
    
    # covert c++ array to numpy array
    cdef unsigned char[:] tmp_skeletons = <unsigned char[:segmentation.size]> cpp_skeletons
    skeletons = np.reshape(np.asarray(tmp_skeletons), grid_size)

    return skeletons



# generate skeletons for this volume
def GenerateTeaserSkeletons(prefix):
    # get the segmentation for this prefix
    segmentation = dataIO.ReadSegmentationData(prefix)

    # get the grid size and world resolution
    world_res = dataIO.Resolution(prefix)

    joined_skeletons = np.zeros(segmentation.shape, dtype=segmentation.dtype)

    import time
    start_time = time.time()

    unique_labels = np.unique(segmentation)
    for label in unique_labels:    
        # skip the zero label if it occurs
        if not label: continue

        if (label > 130): continue

        # perform preprocessing on the segment
        start_time = time.time()
        zmin, ymin, xmin, preprocessed_segmentation = PreprocessSegment(segmentation, label)
        zres, yres, xres = preprocessed_segmentation.shape
        print 'Cropped segment {} in {} seconds'.format(label, time.time() - start_time)
        
        skeletons = IndividualSkeleton(preprocessed_segmentation, world_res)

        joined_skeletons[zmin:zmin+zres,ymin:ymin+yres,xmin:xmin+xres] += label * skeletons

    print time.time() - start_time

    dataIO.WriteH5File(joined_skeletons, 'skeletons-mine.h5', 'main')