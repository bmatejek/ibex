cimport cython
cimport numpy as np
import ctypes
import numpy as np

from ibex.utilities import dataIO
#from ibex.skeletonization.lookup_tables import GenerateLookupTables


cdef extern from 'cpp-generate_skeletons.h':
    long *CppGenerateSkeletons(long *input_segmentation, long input_zres, long input_yres, long input_xres, char *lookup_table_directory)




# generate skeletons for this volume
def GenerateSkeletons(prefix):

    # get the segmentation for this prefix
    segmentation = dataIO.ReadSegmentationData(prefix)
    # add slight padding to the segmentation
    segmentation = np.pad(segmentation, 1, 'constant')


    zres, yres, xres = segmentation.shape


    segmentation[segmentation != 430] = 0
    segmentation[segmentation == 430] = 1

    # read in the array
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef long *cpp_skeletons = CppGenerateSkeletons(&(cpp_segmentation[0,0,0]), zres, yres, xres, "/home/bmatejek/ibex/skeletonization")

    cdef long[:] tmp_skeletons = <long[:segmentation.size]> cpp_skeletons
    skeletons = np.reshape(np.asarray(tmp_skeletons), (zres, yres, xres))

    skeletons = skeletons[1:-1,1:-1,1:-1]

    dataIO.WriteH5File(skeletons, 'skeletons-mine.h5', 'main')

    import sys
    sys.exit()
