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
    void SetOutputDirectory(char *directory)
    void CppTopologicalThinning(const char *prefix, long resolution[3], const char *lookup_table_directory)



# generate skeletons for this volume
def GenerateSkeletons(prefix, resolution=(100, 100, 100)):
    start_time = time.time()

    # get the output directory
    home_directory = os.path.expanduser('~')
    directory = '{}/neuronseg/skeletons/topological-thinning/{}'.format(home_directory, prefix)
    SetOutputDirectory(directory)

    # create the output directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # convert to numpy array for c++ call
    resolution = np.array(resolution, dtype=np.int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_resolution = np.ascontiguousarray(resolution, dtype=ctypes.c_int64)

    # call the topological skeleton algorithm
    CppTopologicalThinning(prefix, &(cpp_resolution[0]), '/home/bmatejek/ibex/skeletonization')

    print '\nSkeletonization time: {}'.format(time.time() - start_time)