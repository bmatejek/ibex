import numpy as np


from ibex.utilities import dataIO
from ibex.transforms import seg2seg, seg2gold
from ibex.cnns.skeleton.util import FindCandidates
from ibex.data_structures import unionfind
from ibex.evaluation import comparestacks
from ibex.transforms import seg2gold



def Oracle(prefix, threshold, maximum_distance, endpoint_distance, network_distance, filtersize=0):
    # get all of the candidates
    positive_candidates = FindCandidates(prefix, threshold, maximum_distance, endpoint_distance, network_distance, 'positive')  
    negative_candidates = FindCandidates(prefix, threshold, maximum_distance, endpoint_distance, network_distance, 'negative')
    candidates = positive_candidates + negative_candidates

    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)

    seg2gold_mapping = seg2gold.Mapping(segmentation, gold)

    # create the union find data structure
    max_value = np.amax(segmentation) + 1
    union_find = [unionfind.UnionFindElement(iv) for iv in range(max_value)]

    # iterate over all candidates and collapse edges
    for candidate in candidates:
        label_one = candidate.labels[0]
        label_two = candidate.labels[1]

        if not seg2gold_mapping[label_one] or not seg2gold_mapping[label_two]: continue
        if (seg2gold_mapping[label_one] == seg2gold_mapping[label_two]):
            unionfind.Union(union_find[label_one], union_find[label_two])

    # create a mapping for the labels
    mapping = np.zeros(max_value, dtype=np.int64)
    for iv in range(max_value):
        mapping[iv] = unionfind.Find(union_find[iv]).label

    segmentation = seg2seg.MapLabels(segmentation, mapping)
    comparestacks.CremiEvaluate(segmentation, gold, dilate_ground_truth=1, mask_ground_truth=True, mask_segmentation=False, filtersize=filtersize)
