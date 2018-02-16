import math
import time
import os

cimport cython
cimport numpy as np
import ctypes
import numpy as np

from ibex.utilities import dataIO
from ibex.skeletonization.lookup_tables import GenerateSmoothingLookupTable
from ibex.utilities.constants import *


cdef extern from 'cpp-generate_skeletons.h':
    void SetDirectory(char *directory)
    void CppTopologicalDownsampleData(long *input_segmentation, long resolution[3], int ratio[3])
    void CppGenerateSkeletons(long label, unsigned char smoothing, unsigned char first_only, char *lookup_table_directory)




# generate skeletons for this volume
def GenerateSkeletons(prefix, ratio, smoothing, first_only):
    start_time = time.time()
    assert (len(ratio) == 3)

    # generate the smoothing lookup table if it is not already there
    GenerateSmoothingLookupTable()

    # needed for cython conversion
    ratio = (int(ratio[IB_Z]), int(ratio[IB_Y]), int(ratio[IB_X]))

    # get the output directory
    home_directory = os.path.expanduser('~')

    str_ratio = '{}x{}x{}'.format(ratio[IB_Z], ratio[IB_Y], ratio[IB_X])

    import sys
    if smoothing and first_only:
        sub_directory = 'with-first-smoothing-{}'.format(str_ratio)
    elif smoothing:
        sub_directory = 'with-complete-smoothing-{}'.format(str_ratio)
    else:
        sub_directory = 'without-smoothing-{}'.format(str_ratio)

    directory = '{}/neuronseg/skeletons/topological-thinning/{}/{}'.format(home_directory, sub_directory, prefix)
    SetDirectory(directory);

    # get the segmentation for this prefix
    segmentation = dataIO.ReadSegmentationData(prefix)

    high_zres, high_yres, high_xres = segmentation.shape
    low_zres, low_yres, low_xres = (int(math.ceil(high_zres / ratio[IB_Z])), int(math.ceil(high_yres / ratio[IB_Y])), int(math.ceil(high_xres / ratio[IB_X])))

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    CppTopologicalDownsampleData(&(cpp_segmentation[0,0,0]), [high_zres, high_yres, high_xres], [ratio[IB_Z], ratio[IB_Y], ratio[IB_X]])

    # get the resolution 
    high_zres, high_yres, high_xres = segmentation.shape
    low_zres, low_yres, low_xres = (int(math.ceil(high_zres / ratio[IB_Z])), int(math.ceil(high_yres / ratio[IB_Y])), int(math.ceil(high_xres / ratio[IB_X])))

    # create the empty skeleton array
    skeletons = np.zeros((low_zres, low_yres, low_xres), dtype=np.int64)

    unique_labels = np.unique(segmentation)
    for label in unique_labels:
        CppGenerateSkeletons(label, int(smoothing), int(first_only), "/home/bmatejek/ibex/skeletonization")

    print 'Skeletonization time: {}'.format(time.time() - start_time)