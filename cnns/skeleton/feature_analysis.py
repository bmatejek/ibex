from ibex.utilities import dataIO
from ibex.cnns.skeleton.util import ExtractFeature, FindCandidates


def FeatureAnalysis(prefix, threshold, maximum_distance):
    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    world_res = dataIO.Resolution(prefix)

    # get the radii for the relevant region
    radii = (maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X])
    width = (2 * radii[IB_Z], 2 * radii[IB_Y], 2 * radii[IB_X])

    # get the candidates
    candidates = FindCandidates(prefix, threshold, maximum_distance, inference=True)
    ncandidates = len(candidates)

    # for each candidate generate the PCA analysis
    for candidate in candidates:
        example = ExtractFeature(segmentation, candidate, width, radii, rotation)
        