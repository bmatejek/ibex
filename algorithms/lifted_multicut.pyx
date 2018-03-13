import numpy as np
import scipy
import math

cimport cython
cimport numpy as np
import ctypes

from util import CollapseGraph, RetrieveCandidates
from ibex.transforms import seg2seg, seg2gold
from ibex.utilities import dataIO
from ibex.evaluation import comparestacks
from ibex.evaluation.classification import *



# c++ external definition
cdef extern from 'cpp-lifted-multicut.h':
    unsigned char *CppLiftedMulticut(unsigned long nvertices, unsigned long nedges, unsigned long *vertex_ones, unsigned long *vertex_twos, double *lifted_weights, double beta, unsigned int heuristic)



def GenerateLiftedEdges(vertex_ones, vertex_twos, edge_weights, nvertices):
    nedges = edge_weights.size
    updated_edge_weights = np.zeros(nedges, dtype=np.float32)
    for ie in range(nedges):
        updated_edge_weights[ie] = -math.log(edge_weights[ie])

    sparse_graph = scipy.sparse.coo_matrix((updated_edge_weights, (vertex_ones, vertex_twos)), shape=(nvertices, nvertices))
    dijkstra_distance = scipy.sparse.csgraph.dijkstra(sparse_graph, directed=False)

    return np.exp(-1 * dijkstra_distance)




def LiftedMulticut(prefix, candidates, edge_weights, beta, threshold, heuristic):
    # read in the segmentation for this prefix
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)

    # get the number of vertices and edges
    nvertices = np.amax(segmentation) + 1
    nedges = edge_weights.size

    # convert the candidate labels to vertices
    vertex_ones = np.zeros(nedges, dtype=np.uint64)
    vertex_twos = np.zeros(nedges, dtype=np.uint64)

    # populate vertex arrays
    for iv, candidate in enumerate(candidates):
        label_one, label_two = candidate.labels
        vertex_ones[iv] = label_one
        vertex_twos[iv] = label_two

    # convert to c++ arrays
    cdef np.ndarray[unsigned long, ndim=1, mode='c'] cpp_vertex_ones = np.ascontiguousarray(vertex_ones, dtype=ctypes.c_uint64)
    cdef np.ndarray[unsigned long, ndim=1, mode='c'] cpp_vertex_twos = np.ascontiguousarray(vertex_twos, dtype=ctypes.c_uint64)

    # generate the lifted edges
    lifted_weights = GenerateLiftedEdges(vertex_ones, vertex_twos, edge_weights, nvertices)
    cdef np.ndarray[double, ndim=2, mode='c'] cpp_lifted_weights = np.ascontiguousarray(lifted_weights, dtype=ctypes.c_double)

    # run multicut algorithm
    cdef unsigned char *cpp_collapsed_edges = CppLiftedMulticut(nvertices, nedges, &(cpp_vertex_ones[0]), &(cpp_vertex_twos[0]), &(cpp_lifted_weights[0,0]), beta, heuristic)
    cdef unsigned char[:] tmp_collapsed_edges = <unsigned char[:nedges]> cpp_collapsed_edges
    maintain_edges = np.asarray(tmp_collapsed_edges).astype(dtype=np.bool)
    
    ncandidates = len(candidates)
    labels = np.zeros(ncandidates, dtype=np.uint8)
    for ie, candidate in enumerate(candidates):
        labels[ie] = candidate.ground_truth

    print 'After Multicut'

    PrecisionAndRecall(labels, 1 - maintain_edges)

    # collapse the edges returned from multicut
    output_filename = 'multicuts/{}-{:0.2f}-lifted.results'.format(prefix, beta)
    CollapseGraph(segmentation, candidates, maintain_edges, edge_weights, output_filename)
   


# function ro run multicut algorithm
def RunLiftedMulticut(prefix, model_prefix, threshold, maximum_distance, endpoint_distance, network_distance, beta, heuristic=1):
    # read the candidates
    candidates, edge_weights = RetrieveCandidates(prefix, model_prefix, threshold, maximum_distance, endpoint_distance, network_distance)

    # run the multicut algorithm
    LiftedMulticut(prefix, candidates, edge_weights, beta, threshold, heuristic)