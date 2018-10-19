cimport cython
cimport numpy as np

import os
import numpy as np
import ctypes


from ibex.graphs.biological.util import ExtractExample, FindSmallSegments, ScaleFeature
from ibex.graphs.biological import edge_generation
from ibex.utilities import dataIO
from ibex.utilities.constants import *



cdef extern from 'cpp-node-generation.h':
    void CppFindMiddleBoundaries(long *segmentation, long grid_size[3])
    void CppGetMiddleBoundaryLocation(long label_one, long label_two, float &zpoint, float &ypoint, float &xpoint)



# simple function to create directory structure for all of the features
def CreateDirectoryStructure(widths, radius, subsets):
    for width in widths:
        # make sure directory structure exists
        directory = 'features/biological/nodes-{}nm-{}x{}x{}'.format(radius, width[IB_Z], width[IB_Y], width[IB_X])
        if not os.path.exists(directory):
            os.mkdir(directory)

        # add all subsets
        for subset in subsets:
            sub_directory = '{}/{}'.format(directory, subset)
            if not os.path.exists(sub_directory):
                os.mkdir(sub_directory)
            # there are three possible labels per subset
            labelings = ['positives', 'negatives', 'unknowns']
            for labeling in labelings:
                if not os.path.exists('{}/{}'.format(sub_directory, labeling)):
                    os.mkdir('{}/{}'.format(sub_directory, labeling))
      



def FindMiddleBoundaries(segmentation):
    # everything needs to be long ints to work with c++
    assert (segmentation.dtype == np.int64)

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_grid_size = np.ascontiguousarray(segmentation.shape, dtype=ctypes.c_int64)

    CppFindMiddleBoundaries(&(cpp_segmentation[0,0,0]), &(cpp_grid_size[0]))

    # free memory
    del cpp_segmentation
    del cpp_grid_size



def GetMiddleBoundary(label_one, label_two):
    cpp_label_one = min(label_one, label_two)
    cpp_label_two = max(label_one, label_two)

    # the center point on the boundary sent to cython
    cdef np.ndarray[float, ndim=1, mode='c'] cpp_point = np.zeros(3, dtype=ctypes.c_float)

    CppGetMiddleBoundaryLocation(label_one, label_two, cpp_point[0], cpp_point[1], cpp_point[2])
    
    return (int(cpp_point[IB_Z]), int(cpp_point[IB_Y]), int(cpp_point[IB_X]))




def GenerateNodes(prefix, segmentation, seg2gold_mapping, subset, threshold=20000, radius=600):
    # possible widths for the neural network
    widths = [(18, 52, 52), (20, 60, 60)]

    # create the directory structure to save the features in
    # forward is needed for training and validation data that is cropped
    CreateDirectoryStructure(widths, radius, ['training', 'validation', 'testing', 'forward'])

    # get the complete adjacency graph shows all neighboring edges
    adjacency_graph = edge_generation.ExtractAdjacencyMatrix(segmentation)

    # get the list of nodes over and under the threshold
    small_segments, large_segments = FindSmallSegments(segmentation, threshold)

    # get the locations around a possible merge
    FindMiddleBoundaries(segmentation)

    # get the radius along each dimensions in terms of voxels
    resolution = dataIO.Resolution(prefix)
    (zradius, yradius, xradius) = (int(radius / resolution[IB_Z]), int(radius / resolution[IB_Y]), int(radius / resolution[IB_X]))
    zres, yres, xres = segmentation.shape

    # crop the subset if it overlaps with testing data
    if subset == 'training' or subset == 'validation':
        ((cropped_zmin, cropped_zmax), (cropped_ymin, cropped_ymax), (cropped_xmin, cropped_xmax)) = dataIO.CroppingBox(prefix)
    elif subset == 'testing':
        ((cropped_zmin, cropped_zmax), (cropped_ymin, cropped_ymax), (cropped_xmin, cropped_xmax)) = ((0, zres), (0, yres), (0, xres))
    else:
        sys.stderr.write('Unrecognized subset: {}'.format(subset))

    for iv, (label_one, label_two) in enumerate(adjacency_graph):
        if (label_one in small_segments) ^ (label_two in small_segments):
            zpoint, ypoint, xpoint = GetMiddleBoundary(label_one, label_two)

            # if the center of the point falls outside the cropped box do not include it in training or validation 
            example_subset = subset
            # however, you allow it for forward inference
            if (zpoint < cropped_zmin or cropped_zmax <= zpoint): example_subset = 'forward'
            if (ypoint < cropped_ymin or cropped_ymax <= ypoint): example_subset = 'forward'
            if (xpoint < cropped_xmin or cropped_xmax <= xpoint): example_subset = 'forward'

            # need to make sure that bounding box does not leave location so sizes are correct
            zmin = max(0, zpoint - zradius)
            ymin = max(0, ypoint - yradius)
            xmin = max(0, xpoint - xradius)
            zmax = min(zres, zpoint + zradius + 1)
            ymax = min(yres, ypoint + yradius + 1)
            xmax = min(xres, xpoint + xradius + 1)

            # create the empty example file with three channels corresponding to the value of segment
            example = np.zeros((2 * zradius + 1, 2 * yradius + 1, 2 * xradius + 1), dtype=np.int32)

            # get the valid location around this point
            segment = ExtractExample(segmentation[zmin:zmax,ymin:ymax,xmin:xmax].copy(), label_one, label_two)

            if example.shape == segment.shape:
                example = segment
            else:
                if zmin == 0: zstart = zradius - zpoint
                else: zstart = 0

                if ymin == 0: ystart = yradius - ypoint
                else: ystart = 0

                if xmin == 0: xstart = xradius - xpoint
                else: xstart = 0

                # the second and third channels are one if the corresponding voxels belong to the individual segments
                example[zstart:zstart+segment.shape[IB_Z],ystart:ystart+segment.shape[IB_Y],xstart:xstart+segment.shape[IB_X]] = segment

            for width in widths:
                # get this subdirectory for this CNN width
                sub_directory = 'features/biological/nodes-{}nm-{}x{}x{}'.format(radius, width[IB_Z], width[IB_Y], width[IB_X])
                scaled_example = ScaleFeature(example, width, label_one, label_two)

                # see if these two segments belong to the same node
                gold_one = seg2gold_mapping[label_one]
                gold_two = seg2gold_mapping[label_two]

                # save the data in the appropriate location
                if gold_one < 1 or gold_two < 1: 
                    output_directory = '{}/{}/unknowns'.format(sub_directory, example_subset)
                elif gold_one == gold_two:
                    output_directory = '{}/{}/positives'.format(sub_directory, example_subset)
                else: 
                    output_directory = '{}/{}/negatives'.format(sub_directory, example_subset)

                output_filename = '{}/{}-{}-{}.h5'.format(output_directory, prefix, label_one, label_two)

                # write this example
                dataIO.WriteH5File(scaled_example, output_filename, 'main')
