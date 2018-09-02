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
    void CppTopologicalThinning(const char *prefix, long skeleton_resolution[3], const char *lookup_table_directory, bool benchmark)
    void CppTeaserSetScale(double input_scale)
    void CppTeaserSetBuffer(long input_buffer)
    void CppTeaserSkeletonization(const char *prefix, long skeleton_resolution[3], bool benchmark)
    void CppAStarSetMaxExpansion(double input_max_expansion)
    void CppApplyUpsampleOperation(const char *prefix, long *input_segmentation, long skeleton_resolution[3], long output_resolution[3], const char *skeleton_algorithm, bool benchmark)
    void CppNaiveUpsampleOperation(const char *prefix, long skeleton_resolution[3], const char *skeleton_algorithm, bool benchmark, double scale, long buffer)


# generate skeletons for this volume
def TopologicalThinning(prefix, skeleton_resolution=(100, 100, 100), benchmark=False, naive=False, astar_max_expansion=1.5):
    if benchmark: input_segmentation = dataIO.ReadGoldData(prefix)
    else: input_segmentation = dataIO.ReadSegmentationData(prefix)

    start_time = time.time()

    # convert the numpy arrays to c++
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_skeleton_resolution = np.ascontiguousarray(skeleton_resolution, dtype=ctypes.c_int64)
    lut_directory = os.path.dirname(__file__)

    # call the topological skeleton algorithm
    CppTopologicalThinning(prefix, &(cpp_skeleton_resolution[0]), lut_directory, benchmark)

    # call the upsampling operation
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_input_segmentation = np.ascontiguousarray(input_segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_output_resolution = np.ascontiguousarray(dataIO.Resolution(prefix), dtype=ctypes.c_int64)
    
    CppNaiveUpsampleOperation(prefix, &(cpp_skeleton_resolution[0]), 'thinning', benchmark, -1, -1)
    for astar_max_expansion in [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]:
        CppAStarSetMaxExpansion(astar_max_expansion) 
        CppApplyUpsampleOperation(prefix, &(cpp_input_segmentation[0,0,0]), &(cpp_skeleton_resolution[0]), &(cpp_output_resolution[0]), 'thinning', benchmark)


    #print 'Topological thinning time for {}: {}'.format((skeleton_resolution[0], skeleton_resolution[1], skeleton_resolution[2]), time.time() - start_time)



# use scipy skeletonization for thinning
def MedialAxis(prefix, skeleton_resolution=(100, 100, 100), benchmark=False, naive=False, astar_max_expansion=1.5):
    if benchmark: input_segmentation = dataIO.ReadGoldData(prefix)
    else: input_segmentation = dataIO.ReadSegmentationData(prefix)

    start_time = time.time()

    # read the downsampled filename
    if benchmark: input_filename = 'benchmarks/skeleton/{}-downsample-{}x{}x{}.bytes'.format(prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z])
    else: input_filename = 'skeletons/{}/downsample-{}x{}x{}.bytes'.format(prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z])
    with open(input_filename, 'rb') as rfd:
        zres, yres, xres, max_label = struct.unpack('qqqq', rfd.read(32))

        running_times = []

        if benchmark: output_filename = 'benchmarks/skeleton/{}-downsample-{}x{}x{}-medial-axis-skeleton.pts'.format(prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z])
        else: output_filename = 'skeletons/{}/downsample-{}x{}x{}-medial-axis-skeleton.pts'.format(prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z]) 
        with open(output_filename, 'wb') as wfd:
            wfd.write(struct.pack('q', zres))
            wfd.write(struct.pack('q', yres))
            wfd.write(struct.pack('q', xres))
            wfd.write(struct.pack('q', max_label))

            # go through all labels
            for label in range(max_label):
                label_time = time.time()
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
                running_times.append(time.time() - label_time)

    if benchmark:
       running_times_filename = 'benchmarks/skeleton/running-times/skeleton-times/{}-{}x{}x{}-medial-axis.bytes'.format(prefix, skeleton_resolution[0], skeleton_resolution[1], skeleton_resolution[2])
       with open(running_times_filename, 'wb') as fd:
        fd.write(struct.pack('q', max_label))
        for label in range(max_label):
            fd.write(struct.pack('d', times[label]))

    # call the upsampling operation
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_skeleton_resolution = np.ascontiguousarray(skeleton_resolution, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_input_segmentation = np.ascontiguousarray(input_segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_output_resolution = np.ascontiguousarray(dataIO.Resolution(prefix), dtype=ctypes.c_int64)
    CppNaiveUpsampleOperation(prefix, &(cpp_skeleton_resolution[0]), 'medial-axis', benchmark, -1, -1)
    for astar_max_expansion in [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]:
        CppAStarSetMaxExpansion(astar_max_expansion) 
        CppApplyUpsampleOperation(prefix, &(cpp_input_segmentation[0,0,0]), &(cpp_skeleton_resolution[0]), &(cpp_output_resolution[0]), 'medial-axis', benchmark)

    #print 'Medial axis thinning time for {}: {}'.format((skeleton_resolution[0], skeleton_resolution[1], skeleton_resolution[2]), time.time() - start_time)



# use TEASER algorithm to generate skeletons
def TEASER(prefix, skeleton_resolution=(100, 100, 100), benchmark=False, teaser_scale=1.3, teaser_buffer=2):
    if benchmark: input_segmentation = dataIO.ReadGoldData(prefix)
    else: input_segmentation = dataIO.ReadSegmentationData(prefix)

    start_time = time.time()

    # convert to numpy array for c++ call
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_skeleton_resolution = np.ascontiguousarray(skeleton_resolution, dtype=ctypes.c_int64)

    # call the teaser skeletonization algorithm
    CppTeaserSetScale(teaser_scale)
    CppTeaserSetBuffer(teaser_buffer)
    CppTeaserSkeletonization(prefix, &(cpp_skeleton_resolution[0]), benchmark)

    # call the upsampling operation
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_input_segmentation = np.ascontiguousarray(input_segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_output_resolution = np.ascontiguousarray(dataIO.Resolution(prefix), dtype=ctypes.c_int64)
    
    CppNaiveUpsampleOperation(prefix, &(cpp_skeleton_resolution[0]), 'teaser', benchmark, teaser_scale, teaser_buffer)

    #print 'TEASER skeletonization time for {}: {}'.format((skeleton_resolution[0], skeleton_resolution[1], skeleton_resolution[2]), time.time() - start_time)