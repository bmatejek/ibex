from ibex.utilities import dataIO
from ibex.transforms import seg2gold


def DetectMergeErrors(prefix):
    # read in the datasets
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)

    # find the mapping from segmentation to gold
    seg2gold_mapping = seg2gold.Mapping(segmentation, gold)

    # read in the skeletons
    skeletons = dataIO.ReadSkeletons(prefix)

    for skeleton in skeletons.skeletons:
        if not skeleton.size(): continue

        print skeleton.label

    print len(skeletons)