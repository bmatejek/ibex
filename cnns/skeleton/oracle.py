import numpy as np


from ibex.utilities import dataIO
from ibex.transforms import seg2seg, seg2gold
from ibex.cnns.skeleton.util import FindCandidates
from ibex.data_structures import unionfind
from PixelPred2Seg import comparestacks



def Oracle(prefix, threshold, maximum_distance, network_distance):
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)

    seg2gold_mapping = seg2gold.Mapping(segmentation, gold, low_threshold=0.10, high_threshold=0.80)

    # connect all segment that have the same gold label
    max_segmentation = np.max(segmentation) + 1
    mapping = np.zeros(max_segmentation, dtype=np.int64)

    nzeros = 1
    for iv in range(max_segmentation):
        if not seg2gold_mapping[iv]:
            mapping[iv] = max_segmentation + nzeros
            nzeros += 1
        else:
            mapping[iv] = seg2gold_mapping[iv]

    oracle_segmentation = seg2seg.MapLabels(segmentation, mapping)
    comparestacks.Evaluate(oracle_segmentation, gold)

    return


    # get all of the candidates
    candidates = FindCandidates(prefix, threshold, maximum_distance, network_distance, inference=True)  
    ncandidates = len(candidates)

    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, threshold=threshold)
    gold = dataIO.ReadGoldData(prefix)

    # create the union find data structure
    max_value = np.amax(segmentation) + 1
    union_find = [unionfind.UnionFindElement(iv) for iv in range(max_value)]

    # iterate over all candidates and collapse edges
    for candidate in candidates:
        if not candidate.ground_truth: continue

        label_one = candidate.labels[0]
        label_two = candidate.labels[1]

        unionfind.Union(union_find[label_one], union_find[label_two])

    # create a mapping for the labels
    mapping = np.zeros(max_value, dtype=np.int64)
    for iv in range(max_value):
        mapping[iv] = unionfind.Find(union_find[iv]).label

    # output the mapping
    with open('tmp.txt', 'w') as fd:
        fd.write('{}\n'.format(max_value))
        for iv in range(max_value):
            fd.write('{}\n'.format(mapping[iv]))


    #comparestacks.Evaluate(segmentation, gold, filtersize=threshold, anisotropic=anisotropy)
