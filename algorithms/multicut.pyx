from ibex.transforms import seg2seg
from ibex.data_structures import UnionFind
from ibex.utilities import dataIO
cimport cython
cimport numpy as np
import ctypes
import numpy as np
import struct
import os
import time

cdef extern from 'cpp-multicut.h':
    unsigned char *CppMulticut(unsigned long nvertices, unsigned long nedges, unsigned long *vertex_ones, unsigned long *vertex_twos, double *edge_weights, double threshold, int algorithm)

def CollapseGraph(prefix, collapsed_edges, vertex_ones, vertex_twos):
    start_time = time.time()

    # read the segmentation data
    segmentation = dataIO.ReadSegmentationData(prefix)

    # get the mapping to a smaller set of vertices
    _, reverse_mapping = seg2seg.ReduceLabels(segmentation)

    # get the number of vertices and edges
    nvertices = len(reverse_mapping)
    nedges = vertex_ones.size
    assert (vertex_ones.size == vertex_twos.size)

    # get the maximum value for the segmentation
    max_value = np.uint64(np.amax(segmentation) + 1)

    # create an empty union find data structure
    union_find = [UnionFind.UnionFindElement(iv) for iv in range(max_value)]

    # read all of the labels and merge the result
    for ie in range(nedges):
        # only if this edge should collapse
        if collapsed_edges[ie]: continue

        # get the original labels
        label_one = reverse_mapping[vertex_ones[ie]]
        label_two = reverse_mapping[vertex_twos[ie]]

        # merge label one and two in the union find data structure
        UnionFind.Union(union_find[label_one], union_find[label_two])

    # create a mapping
    mapping = np.zeros(max_value, dtype=np.uint64)

    # update the segmentation
    for iv in range(max_value):
        label = UnionFind.Find(union_find[iv]).label

        mapping[iv] = label

    # update the labels
    segmentation = seg2seg.MapLabels(segmentation, mapping)

    print 'Collapsed graph in {} seconds'.format(time.time() - start_time)

    # return the updated segmentation
    return segmentation

def EvaluateMulticut(prefix, multicut_segmentation, threshold):
    start_time = time.time()

    # TODO fix this code temporary filename
    multicut_filename = 'multicut/{}-multicut.h5'.format(prefix)

    # temporary - write h5 file
    dataIO.WriteH5File(multicut_segmentation, multicut_filename, 'stack')

    # # get the gold filename
    gold_filename = 'gold/{}_gold.h5'.format(prefix)
    # segmentation_filename = 'rhoana/{}_rhoana_stack.h5'.format(prefix)

    # print 'Before multicut: '
    # # create the command line 
    # command = '~/software/PixelPred2Seg/comparestacks --stack1 {} --stackbase {} --dilate1 1 --dilatebase 1 --relabel1 --relabelbase --filtersize 100 --anisotropic'.format(segmentation_filename, gold_filename)

    # # execute the command
    # os.system(command)

    print 'After multicut - {}: '.format(threshold)
    # create the command line 
    command = '~/software/PixelPred2Seg/comparestacks --stack1 {} --stackbase {} --dilate1 1 --dilatebase 1 --relabel1 --relabelbase --filtersize 100 --anisotropic'.format(multicut_filename, gold_filename)

    # execute the command
    os.system(command)

    print 'Evaluated multicut in {} seconds'.format(time.time() - start_time)

def Multicut(prefix, model_prefix, threshold=0.5, algorithm=0):
    start_time = time.time()

    multicut_filename = 'multicut/{}-{}.graph'.format(model_prefix, prefix)

    # open the binary file
    with open(multicut_filename, 'rb') as fd:
        # read the number of vertices and edges
        nvertices, nedges, = struct.unpack('QQ', fd.read(16))

        # create an array for all of the labels
        vertex_ones = np.zeros(nedges, dtype=np.uint64)
        vertex_twos = np.zeros(nedges, dtype=np.uint64)
        edge_weights = np.zeros(nedges, dtype=np.float64)

        # read in values for all of the edges
        for ie in range(nedges):
            # skip over the original labels - not needed
            fd.read(16)

            # read in the vertices connected by this edge
            vertex_ones[ie], vertex_twos[ie], edge_weights[ie], = struct.unpack('QQd', fd.read(24))

    # call the multicut cpp function
    cdef np.ndarray[unsigned long, ndim=1, mode='c'] cpp_vertex_ones
    cpp_vertex_ones = np.ascontiguousarray(vertex_ones, dtype=ctypes.c_uint64)
    cdef np.ndarray[unsigned long, ndim=1, mode='c'] cpp_vertex_twos
    cpp_vertex_twos = np.ascontiguousarray(vertex_twos, dtype=ctypes.c_uint64)
    cdef np.ndarray[double, ndim=1, mode='c'] cpp_edge_weights 
    cpp_edge_weights = np.ascontiguousarray(edge_weights, dtype=ctypes.c_double)

    cdef unsigned char *cpp_collapsed_edges = CppMulticut(nvertices, nedges, &(cpp_vertex_ones[0]), &(cpp_vertex_twos[0]), &(cpp_edge_weights[0]), threshold, algorithm)
    cdef unsigned char[:] tmp_collapsed_edges = <unsigned char[:nedges]> cpp_collapsed_edges
    collapsed_edges = np.asarray(tmp_collapsed_edges).astype(dtype=np.bool)

    print 'Ran multicut in {} seconds'.format(time.time() - start_time)

    # collapse the edges
    multicut_segmentation = CollapseGraph(prefix, collapsed_edges, vertex_ones, vertex_twos)

    EvaluateMulticut(prefix, multicut_segmentation, threshold)

