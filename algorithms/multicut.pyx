import numpy as np

cimport cython
cimport numpy as np
import ctypes

from util import CollapseGraph, RetrieveCandidates
from ibex.transforms import seg2seg, seg2gold
from ibex.utilities import dataIO
from ibex.evaluation import comparestacks
from ibex.evaluation.classification import *



# c++ external definition
cdef extern from 'cpp-multicut.h':
    unsigned char *CppMulticut(unsigned long nvertices, unsigned long nedges, unsigned long *vertex_ones, unsigned long *vertex_twos, double *edge_weights, double beta, unsigned int heuristic)



def Multicut(prefix, candidates, edge_weights, beta, threshold, heuristic):
    # read in the segmentation for this prefix
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)

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
    cdef unsigned char *cpp_collapsed_edges = CppMulticut(nvertices, nedges, &(cpp_vertex_ones[0]), &(cpp_vertex_twos[0]), &(cpp_edge_weights[0]), beta, heuristic)
    cdef unsigned char[:] tmp_collapsed_edges = <unsigned char[:nedges]> cpp_collapsed_edges
    maintain_edges = np.asarray(tmp_collapsed_edges).astype(dtype=np.bool)
    
    ncandidates = len(candidates)
    labels = np.zeros(ncandidates, dtype=np.uint8)
    for ie, candidate in enumerate(candidates):
        labels[ie] = candidate.ground_truth

    print '\nAfter Multicut\n'

    PrecisionAndRecall(labels, 1 - maintain_edges)

    # collapse the edges returned from multicut
    output_filename = 'multicuts/{}-{}.results'.format(prefix, beta)
    CollapseGraph(segmentation, candidates, maintain_edges, edge_weights, output_filename)
    




# function ro run multicut algorithm
def RunMulticut(prefix, model_prefix, threshold, maximum_distance, endpoint_distance, network_distance, beta, heuristic=1):
    # read the candidates
    candidates, edge_weights = RetrieveCandidates(prefix, model_prefix, threshold, maximum_distance, endpoint_distance, network_distance)

    # run the multicut algorithm
    Multicut(prefix, candidates, edge_weights, beta, threshold, heuristic)