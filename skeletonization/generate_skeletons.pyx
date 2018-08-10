import math
import time
import os
import struct
import inspect


cimport cython
cimport numpy as np
from libcpp cimport bool
import ctypes
import numpy as np
import skimage.morphology

from ibex.utilities import dataIO
from ibex.utilities.constants import *
from medial_axis_util import PostProcess


cdef extern from 'cpp-generate_skeletons.h':
    void CppTopologicalThinning(const char *prefix, long resolution[3], const char *lookup_table_directory, bool benchmark)
    void CppTeaserSkeletonization(const char *prefix, long resolution[3], bool benchmark)
    void CppApplyUpsampleOperation(const char *prefix, long resolution[3], const char *skeleton_algorithm, bool benchmark)


# generate skeletons for this volume
def TopologicalThinning(prefix, resolution=(100, 100, 100), benchmark=False):
    start_time = time.time()

    # convert to numpy array for c++ call
    resolution = np.array(resolution, dtype=np.int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_resolution = np.ascontiguousarray(resolution, dtype=ctypes.c_int64)

    print inspect.stack()[0][1]

    # call the topological skeleton algorithm
    CppTopologicalThinning(prefix, &(cpp_resolution[0]), '/home/bmatejek/ibex/skeletonization', benchmark)

    print 'Topological thinning time for {}: {}'.format((resolution[0], resolution[1], resolution[2]), time.time() - start_time)



# use scipy skeletonization for thinning
def MedialAxis(prefix, resolution=(100, 100, 100), benchmark=False):
    start_time = time.time()

    # read the downsampled filename
    if benchmark: input_filename = 'benchmarks/skeleton/{}-topological-downsample-{}x{}x{}.bytes'.format(prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z])
    else: input_filename = 'skeletons/{}/topological-downsample-{}x{}x{}.bytes'.format(prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z])
    with open(input_filename, 'rb') as rfd:
        zres, yres, xres, max_label = struct.unpack('qqqq', rfd.read(32))

        if benchmark: output_filename = 'benchmarks/skeleton/{}-topological-downsample-{}x{}x{}-medial-axis-skeleton.pts'.format(prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z])
        else: output_filename = 'skeletons/{}/topological-downsample-{}x{}x{}-medial-axis-skeleton.pts'.format(prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]) 
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

                skeleton = PostProcess(skimage.morphology.skeletonize_3d(segmentation))

                nelements = len(skeleton)
                wfd.write(struct.pack('q', nelements))
                for element in skeleton:
                    wfd.write(struct.pack('q', element))

    # upsample these skeletons
    resolution = np.array(resolution, dtype=np.int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_resolution = np.ascontiguousarray(resolution, dtype=ctypes.c_int64)

    CppApplyUpsampleOperation(prefix, &(cpp_resolution[0]), "medial-axis", benchmark)

    print 'Medial axis thinning time for {}: {}'.format((resolution[0], resolution[1], resolution[2]), time.time() - start_time)



# use TEASER algorithm to generate skeletons
def TeaserSkeletonization(prefix, resolution=(100, 100, 100), benchmark=False):
    start_time = time.time()

    # convert to numpy array for c++ call
    resolution = np.array(resolution, dtype=np.int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_resolution = np.ascontiguousarray(resolution, dtype=ctypes.c_int64)

    # call the teaser skeletonization algorithm
    CppTeaserSkeletonization(prefix, &(cpp_resolution[0]), benchmark)

    print 'TEASER skeletonization time for {}: {}'.format((resolution[0], resolution[1], resolution[2]), time.time() - start_time)