import math
import time
import os
import struct

cimport cython
cimport numpy as np
import ctypes
import numpy as np
import skimage.morphology

from ibex.utilities import dataIO
from ibex.utilities.constants import *


cdef extern from 'cpp-generate_skeletons.h':
    void CppTopologicalThinning(const char *prefix, long resolution[3], const char *lookup_table_directory)
    void CppTeaserSkeletonization(const char *prefix, long resolution[3])


# generate skeletons for this volume
def TopologicalThinning(prefix, resolution=(100, 100, 100)):
    start_time = time.time()

    # convert to numpy array for c++ call
    resolution = np.array(resolution, dtype=np.int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_resolution = np.ascontiguousarray(resolution, dtype=ctypes.c_int64)

    # call the topological skeleton algorithm
    CppTopologicalThinning(prefix, &(cpp_resolution[0]), '/home/bmatejek/ibex/skeletonization')

    print '\nTopological thinning time: {}'.format(time.time() - start_time)



# use scipy skeletonization for thinning
def MedialAxis(prefix, resolution=(100, 100, 100)):
    start_time = time.time()

    # read the downsampled filename
    input_filename = 'topological/{}-topological-downsample-{}x{}x{}.bytes'.format(prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z])
    with open(input_filename, 'rb') as rfd:
        zres, yres, xres, max_label = struct.unpack('qqqq', rfd.read(32))

        output_filename = 'topological/{}-topological-downsample-{}x{}x{}-medial-axis-skeleton.pts'.format(prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z])
        with open(output_filename, 'wb') as wfd:
            wfd.write(struct.pack('q', max_label))

            # go through all labels
            for label in range(max_label):
                segmentation = np.zeros((zres, yres, xres), dtype=np.bool)

                # find topological downsampled locations
                nelements, = struct.unpack('q', rfd.read(8))
                for _ in range(nelements):
                    iv, = struct.unpack('q', rfd.read(8))

                    iz = iv / (xres * yres)
                    iy = (iv - iz * xres * yres) / xres
                    ix = iv % xres

                    segmentation[iz,iy,ix] = 1

                skeleton = np.nonzero(skimage.morphology.skeletonize_3d(segmentation).flatten())[0]
                nelements = len(skeleton)
                wfd.write(struct.pack('q', nelements))
                for element in skeleton:
                    wfd.write(struct.pack('q', element))


    print 'Medial axis thinning time: {}'.format(time.time() - start_time)



# use TEASER algorithm to generate skeletons
def TeaserSkeletonization(prefix, resolution=(100, 100, 100)):
    start_time = time.time()

    # convert to numpy array for c++ call
    resolution = np.array(resolution, dtype=np.int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_resolution = np.ascontiguousarray(resolution, dtype=ctypes.c_int64)

    # call the teaser skeletonization algorithm
    CppTeaserSkeletonization(prefix, &(cpp_resolution[0]))

    print 'TEASER skeletonization time: {}'.format(time.time() - start_time)