cimport cython
cimport numpy as np
import ctypes
import numpy as np
import time
import struct

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
    # read the topological files
    input_topological_filename = 'topological/{}-topological-downsample-100x100x100.bytes'.format(prefix)
    output_topological_filename = 'topological/{}-topological-downsample-100x100x100-skeleton.bytes'.format(prefix)
    with open(input_topological_filename, 'rb') as rd:
        with open(output_topological_filename, 'wb') as wd:
            zres, yres, xres, max_segment, = struct.unpack('qqqq', rd.read(32))
            wd.write(struct.pack('qqqq', zres, yres, xres, max_segment))

            for segment in range(max_segment):
                segmentation = np.zeros((zres, yres, xres), dtype=np.int64)

                nelements, = struct.unpack('q', rd.read(8))

                for _ in range(nelements):
                    iv, = struct.unpack('q', rd.read(8))

                    iz = iv / (yres * xres)
                    iy = (iv - iz * yres * xres) / xres
                    ix = iv % xres

                    segmentation[iz,iy,ix] = 1

                skeleton = IndividualSkeleton(segmentation, (1,1,1))
                nelements = np.count_nonzero(skeleton)
                wd.write(struct.pack('q', nelements))

                for iz in range(zres):
                    for iy in range(yres):
                        for ix in range(xres):
                            if skeleton[iz,iy,ix]:
                                wd.write(struct.pack('q', iz * yres * xres + iy * xres + ix))

                if segment == 200: break




    # # get the segmentation for this prefix
    # segmentation = dataIO.ReadSegmentationData(prefix)

    # # get the grid size and world resolution
    # world_res = dataIO.Resolution(prefix)

    # joined_skeletons = np.zeros(segmentation.shape, dtype=segmentation.dtype)

    # import time
    # start_time = time.time()

    # unique_labels = np.unique(segmentation)
    # for label in unique_labels:    
    #     # skip the zero label if it occurs
    #     if not label: continue

    #     if not label == 19: continue

    #     # perform preprocessing on the segment
    #     start_time = time.time()
    #     zmin, ymin, xmin, preprocessed_segmentation = PreprocessSegment(segmentation, label)
    #     zres, yres, xres = preprocessed_segmentation.shape
    #     print 'Cropped segment {} in {} seconds'.format(label, time.time() - start_time)
        
    #     skeletons = IndividualSkeleton(preprocessed_segmentation, world_res)

    #     joined_skeletons[zmin:zmin+zres,ymin:ymin+yres,xmin:xmin+xres] += label * skeletons

    # print time.time() - start_time

    # dataIO.WriteH5File(joined_skeletons, 'skeletons-mine.h5', 'main')