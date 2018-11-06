import struct
import numpy as np


from ibex.transforms import seg2gold, seg2seg
from ibex.data_structures import unionfind
from ibex.evaluation.classification import *
from ibex.utilities import dataIO



def PrintResults(prefix, vertex_ones, vertex_twos, edge_weights, maintained_edges):
    # get the ground truth and print out the results
    seg2gold_mapping = seg2gold.Mapping(prefix)

    # get the number of edges
    nedges = edge_weights.shape[0]

    # see how multicut has changed the results
    labels = []
    cnn_results = []
    multicut_results = []

    # go through each edge
    for ie in range(nedges):
        vertex_one = vertex_ones[ie]
        vertex_two = vertex_twos[ie]

        # skip if there is no ground truth
        if seg2gold_mapping[vertex_one] < 1 or seg2gold_mapping[vertex_two] < 1: continue

        # over 0.5 on edge weight means the edge should collapse
        cnn_results.append(edge_weights[ie] > 0.5)

        # since this edge has ground truth add to list
        # subtract one here since a maintained edge is one that should not be merged
        multicut_results.append(1 - maintained_edges[ie])

        if seg2gold_mapping[vertex_one] == seg2gold_mapping[vertex_two]: labels.append(True)
        else: labels.append(False)

    print 'CNN Results:'
    PrecisionAndRecall(np.array(labels), np.array(cnn_results))
    print 'Multicut Results'
    PrecisionAndRecall(np.array(labels), np.array(multicut_results))



def ReadCandidates(prefix, model_prefix):
    # get the input file with all of the probabilities
    input_filename = '{}-{}.probabilities'.format(model_prefix, prefix)

    # read all of the candidates and probabilities
    with open(input_filename, 'rb') as fd:
        nexamples, = struct.unpack('q', fd.read(8))

        vertex_ones = np.zeros(nexamples, dtype=np.int64)
        vertex_twos = np.zeros(nexamples, dtype=np.int64)
        edge_weights = np.zeros(nexamples, dtype=np.float64)

        for ie in range(nexamples):
            vertex_ones[ie], vertex_twos[ie], edge_weights[ie], = struct.unpack('qqd', fd.read(24))

    # return the list of vertices and corresponding probabilities
    return vertex_ones, vertex_twos, edge_weights



def CollapseGraph(prefix, segmentation, vertex_ones, vertex_twos, maintained_edges, algorithm):
    # get the number of edges
    nedges = maintained_edges.shape[0]

    # create the union find data structure and collapse the graph
    max_label = np.amax(segmentation) + 1
    union_find = [unionfind.UnionFindElement(iv) for iv in range(max_label)]

    # go through all of the edges
    for ie in range(nedges):
        # skip if the edge should not collapse
        if maintained_edges[ie]: continue

        # merge these vertices
        vertex_one = vertex_ones[ie]
        vertex_two = vertex_twos[ie]

        unionfind.Union(union_find[vertex_one], union_find[vertex_two])

    # create the mapping and save the result
    mapping = np.zeros(max_label, dtype=np.int64)
    for iv in range(max_label):
        mapping[iv] = unionfind.Find(union_find[iv]).label

    # apply the mapping and save the result
    seg2seg.MapLabels(segmentation, mapping)

    output_filename = 'rhoana/{}-{}.h5'.format(prefix, algorithm)
    dataIO.WriteH5File(segmentation, output_filename, 'main')

# import struct
# import numpy as np

# import ibex.cnns.skeleton.util
# from ibex.evaluation.classification import *
# from ibex.data_structures import unionfind
# from ibex.transforms import seg2seg
# from ibex.utilities import dataIO
# from ibex.evaluation import comparestacks



# def RetrieveCandidates(prefix, model_prefix, threshold, maximum_distance, endpoint_distance, network_distance):
#     # get all of the candidates for this brain
#     positive_candidates = ibex.cnns.skeleton.util.FindCandidates(prefix, threshold, maximum_distance, endpoint_distance, network_distance, 'positive')
#     negative_candidates = ibex.cnns.skeleton.util.FindCandidates(prefix, threshold, maximum_distance, endpoint_distance, network_distance, 'negative')
#     undetermined_candidates = ibex.cnns.skeleton.util.FindCandidates(prefix, threshold, maximum_distance, endpoint_distance, network_distance, 'undetermined')
    
#     candidates = positive_candidates + negative_candidates + undetermined_candidates
#     ncandidates = len(candidates)

#     # read the probabilities foor this candidate
#     probabilities_filename = '{}-{}-{}-{}nm-{}nm-{}nm.probabilities'.format(model_prefix, prefix, threshold, maximum_distance, endpoint_distance, network_distance)
#     with open(probabilities_filename, 'rb') as fd:
#         nprobabilities, = struct.unpack('i', fd.read(4))
#         assert (nprobabilities == ncandidates)
#         edge_weights = np.zeros(nprobabilities, dtype=np.float64)
#         for iv in range(nprobabilities):
#             edge_weights[iv], = struct.unpack('d', fd.read(8))

#     return candidates, edge_weights



# # collapse the edges from multicut
# def CollapseGraph(segmentation, candidates, maintain_edges, probabilities, output_filename):
#     ncandidates = len(candidates)

#     # get the ground truth and the predictions
#     labels = np.zeros(ncandidates, dtype=np.bool)
#     for iv in range(ncandidates):
#         labels[iv] = candidates[iv].ground_truth

#     # create an empty union find data structure
#     max_value = np.amax(segmentation) + 1
#     union_find = [unionfind.UnionFindElement(iv) for iv in range(max_value)]

#     # create adjacency sets for the elements in the segment
#     adjacency_sets = [set() for _ in range(max_value)]

#     for candidate in candidates:
#         label_one = candidate.labels[0]
#         label_two = candidate.labels[1]

#         adjacency_sets[label_one].add(label_two)
#         adjacency_sets[label_two].add(label_one)

#     # iterate over the candidates in order of decreasing probability
#     zipped = zip(probabilities, [ie for ie in range(ncandidates)])

#     for probability, ie in sorted(zipped, reverse=True):
#         # skip if the edge is not collapsed
#         if maintain_edges[ie]: continue
#         # skip if this creates a cycle
#         label_one, label_two = candidates[ie].labels

#         # get the parent of this label
#         label_two_union_find = unionfind.Find(union_find[label_two]).label

#         # make sure none of the other adjacent nodes already has this label
#         for neighbor_label in adjacency_sets[label_one]:
#             if neighbor_label == label_two: continue

#         if unionfind.Find(union_find[neighbor_label]).label == label_two_union_find: 
#             maintain_edges[ie] = True

#         # skip if the edge is no longer collapsed
#         if maintain_edges[ie]: continue
#         unionfind.Union(union_find[label_one], union_find[label_two])

#     print '\nBorder Constraints\n'
#     PrecisionAndRecall(labels, 1 - maintain_edges)

#     # for every edge, save if the edge is collapsed
#     with open(output_filename, 'wb') as fd:
#         fd.write(struct.pack('q', ncandidates))
#         for ie in range(ncandidates):
#             fd.write(struct.pack('?', maintain_edges[ie]))

#     mapping = np.zeros(max_value, dtype=np.int64)
#     for iv in range(max_value):
#         mapping[iv] = unionfind.Find(union_find[iv]).label

#     segmentation = seg2seg.MapLabels(segmentation, mapping)
#     gold = dataIO.ReadGoldData('SNEMI3D_train')
#     print comparestacks.adapted_rand(segmentation, gold, all_stats=False, dilate_ground_truth=2, filtersize=0)
