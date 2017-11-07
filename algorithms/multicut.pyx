import struct
import numpy as np
import os

cimport cython
cimport numpy as np
import ctypes

import ibex.cnns.skeleton.util
from ibex.transforms import seg2seg
from ibex.utilities import dataIO
from ibex.data_structures import unionfind
from ibex.evaluation.classification import *
from PixelPred2Seg import comparestacks



# c++ external definition
cdef extern from 'cpp-multicut.h':
    unsigned char *CppMulticut(unsigned long nvertices, unsigned long nedges, unsigned long *vertex_ones, unsigned long *vertex_twos, double *edge_weights, double beta)



# collapse the edges from multicut
def CollapseGraph(segmentation, candidates, collapsed_edges, probabilities):
    # read the candidates
    ncandidates = len(candidates)

    # get the ground truth and the predictions
    ground_truth = np.zeros(ncandidates, dtype=np.bool)
    predictions = np.zeros(ncandidates, dtype=np.bool)
    for iv in range(ncandidates):
        ground_truth[iv] = candidates[iv].ground_truth
        predictions[iv] = 1 - collapsed_edges[iv]
        #if predictions[iv] and not ground_truth[iv]:
        #    print '{} {} {}'.format(iv, candidates[iv].labels, probabilities[iv])

    # output the results for multicut
    PrecisionAndRecall(ground_truth, predictions)
    
    #import sys
    #sys.exit()
    
    # create an empty union find data structure
    max_value = np.uint64(np.amax(segmentation) + 1)
    union_find = [unionfind.UnionFindElement(iv) for iv in range(max_value)]

    # iterate over all collapsed edges
    for ie in range(ncandidates):
        # collpased edges is zero where the edge should no longer exist
        if collapsed_edges[ie]: continue

        label_one, label_two = candidates[ie].labels

        unionfind.Union(union_find[label_one], union_find[label_two])

    # create a mapping for the labels
    mapping = np.zeros(max_value, dtype=np.uint64)
    for iv in range(max_value):
        mapping[iv] = unionfind.Find(union_find[iv]).label

    segmentation = seg2seg.MapLabels(segmentation, mapping)

    return segmentation



def Multicut(segmentation, gold, candidates, edge_weights, beta, threshold, anisotropic):
    # get a forward and reverse mapping
    forward_mapping, reverse_mapping = seg2seg.ReduceLabels(segmentation)

    # get the number of vertices and edges
    nvertices = reverse_mapping.size
    nedges = edge_weights.size

    # convert the candidate labels to vertices
    vertex_ones = np.zeros(nedges, dtype=np.uint64)
    vertex_twos = np.zeros(nedges, dtype=np.uint64)

    # populate vertex arrays
    for iv, candidate in enumerate(candidates):
        label_one, label_two = candidate.labels
        vertex_ones[iv] = forward_mapping[label_one]
        vertex_twos[iv] = forward_mapping[label_two]

    # convert to c++ arrays
    cdef np.ndarray[unsigned long, ndim=1, mode='c'] cpp_vertex_ones = np.ascontiguousarray(vertex_ones, dtype=ctypes.c_uint64)
    cdef np.ndarray[unsigned long, ndim=1, mode='c'] cpp_vertex_twos = np.ascontiguousarray(vertex_twos, dtype=ctypes.c_uint64)
    cdef np.ndarray[double, ndim=1, mode='c'] cpp_edge_weights = np.ascontiguousarray(edge_weights, dtype=ctypes.c_double)

    # run multicut algorithm
    cdef unsigned char *cpp_collapsed_edges = CppMulticut(nvertices, nedges, &(cpp_vertex_ones[0]), &(cpp_vertex_twos[0]), &(cpp_edge_weights[0]), beta)
    cdef unsigned char[:] tmp_collapsed_edges = <unsigned char[:nedges]> cpp_collapsed_edges
    collapsed_edges = np.asarray(tmp_collapsed_edges).astype(dtype=np.bool)

    # collapse the edges returned from multicut
    segmentation = CollapseGraph(segmentation, candidates, collapsed_edges, edge_weights)

    # evaluate before and after multicut
    comparestacks.Evaluate(segmentation, gold, threshold, anisotropic)



# function ro run multicut algorithm
def RunMulticut(prefix, model_prefix, threshold, maximum_distance, network_distance, beta, anisotropic=True):
    # read the candidates
    candidates = ibex.cnns.skeleton.util.FindCandidates(prefix, threshold, maximum_distance, network_distance, inference=True)
    ncandidates = len(candidates)
    
    # read the probabilities
    probabilities_filename = '{}-{}-{}-{}nm.probabilities'.format(model_prefix, prefix, threshold, maximum_distance)
    with open(probabilities_filename, 'rb') as fd:
        nprobabilities, = struct.unpack('i', fd.read(4))
        assert (nprobabilities == ncandidates)
        edge_weights = np.zeros(nprobabilities, dtype=np.float64)
        for iv in range(nprobabilities):
            edge_weights[iv], = struct.unpack('d', fd.read(8))
    
    # read in the segmentation for this prefix
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)

    # run the multicut algorithm
    Multicut(segmentation, gold, candidates, edge_weights, beta, threshold, anisotropic)
