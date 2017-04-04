import sys
import struct
import numpy as np
from ibex.transforms import seg2seg
from ibex.utilities import dataIO
from ibex.data_structures import UnionFind
from ibex.evaluation import rhoana_evaluation

def PrintEvaluation(acc):
    print "Rand: correct merge = {0}, correct split = {1}".format(acc['Rand']['merge'],acc['Rand']['split'])
    print "VI: correct merge = {0}, correct split = {1}".format(acc['VI']['merge'],acc['VI']['split'])


# merge the results of multicut
def MergeMulticut(prefix, maximum_distance):
    # read the segmentation data
    segmentation = dataIO.ReadSegmentationData(prefix)

    # get the gold data
    gold = dataIO.ReadGoldData(prefix)

    # determine the starting VI values
    PrintEvaluation(rhoana_evaluation.segmentation_metrics(gold, segmentation))

    # get the mapping to a smaller set of vertices
    forward_mapping, reverse_mapping = seg2seg.ReduceLabels(segmentation)

    # read in the two files
    multicut_filename = 'multicut/{0}_skeleton_{1}nm.graph'.format(prefix, maximum_distance)    
    merge_filename = 'multicut/{0}_multicut_output.graph'.format(prefix)

    # open the files
    multicut_fd = open(multicut_filename, 'rb')
    merge_fd = open(merge_filename, 'rb')

    # read the headers
    nvertices, nedges, = struct.unpack('QQ', multicut_fd.read(16))
    assert (nvertices == len(reverse_mapping))

    nmerge_edges, = struct.unpack('Q', merge_fd.read(8))
    assert (nmerge_edges == nedges)

    # get the maximum value for the segmentation
    max_value = np.amax(segmentation) + 1

    # create empty union find structure
    union_find = [UnionFind.UnionFindElement(iv) for iv in range(max_value)]

    # read all of the labels and the merge result
    for _ in range(nedges):
        reduced_label_one, reduced_label_two, _, = struct.unpack('QQd', multicut_fd.read(24))
        edge_label, = struct.unpack('c', merge_fd.read(1))

        # ignore edges that should not be merged
        if not edge_label: continue

        # get the original labels
        label_one = reverse_mapping[reduced_label_one]
        label_two = reverse_mapping[reduced_label_two]

        # merge label one and two in the union find data structure
        UnionFind.Union(union_find[label_one], union_find[label_two])

    mapping = np.zeros(max_value, dtype=np.uint64)

    # update the segmentation
    for iv in range(max_value):
        label = UnionFind.Find(union_find[iv]).label

        mapping[iv] = label

    # update the labels
    segmentation = seg2seg.MapLabels(segmentation, mapping)

    PrintEvaluation(rhoana_evaluation.segmentation_metrics(gold, segmentation))