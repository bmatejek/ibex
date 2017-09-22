import numpy as np
import random
import struct

from scipy.spatial import KDTree

from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.transforms import seg2seg, seg2gold
from ibex.cnns.skeleton.util import SkeletonCandidate


# create a kdtree from skeleton endpoints
def GenerateKDTree(endpoints, world_res):
    # get the number of endpoints
    npoints = len(endpoints)

    # create an array for locations
    locations = np.zeros((npoints, NDIMS), dtype=np.float32)

    # iterate through all endpoints
    for iv, endpoint in enumerate(endpoints):
        locations[iv] = endpoint.WorldPoint(world_res)

    return locations, KDTree(locations)




# further restrict the locations for candidates
def PruneNeighbors(neighbors, endpoints, radii, grid_size):
    # create an array for merge locations
    pruned_neighbors = []

    # consider all pairs of neighbors
    for neighbor_one, neighbor_list in enumerate(neighbors):
        for neighbor_two in neighbor_list:
            # to avoid double counting neighbors
            if (neighbor_one >= neighbor_two): continue

            # avoid neighbors belonging to the same segment
            if (endpoints[neighbor_one].Label() == endpoints[neighbor_two].Label()): continue

            # get the location for these two neighbors
            point_one = endpoints[neighbor_one].GridPoint()
            point_two = endpoints[neighbor_two].GridPoint()

            # get the midpoint for this location
            midpoint = (point_one + point_two) / 2

            # if this location extends past boundary ignore
            interior_neighbor = True
            for dim in range(NDIMS):
                if (midpoint[dim] - radii[dim] < 0): interior_neighbor = False
                if (midpoint[dim] + radii[dim] >= grid_size[dim]): interior_neighbor = False

            if not interior_neighbor: continue

            # append these neighbors to the 
            pruned_neighbors.append((neighbor_one, neighbor_two))

    return pruned_neighbors



# create the skeleton merge candidate
def GenerateCandidates(neighbors, endpoints, segmentation, gold, seg2gold_mapping, radii):
    positive_candidates = []
    negative_candidates = []
    undetermined_candidates = []

    # iterate through all of the neighbors
    for ie, (neighbor_one, neighbor_two) in enumerate(neighbors):
        # get the label for these neighbors
        label_one = endpoints[neighbor_one].Label()
        label_two = endpoints[neighbor_two].Label()

        # get the location of both endpoints
        point_one = endpoints[neighbor_one].GridPoint()
        point_two = endpoints[neighbor_two].GridPoint()

        # get the midpoint between the endpoints
        midpoint = (point_one + point_two) / 2

        # should these neighbors merge?
        #ground_truth = (seg2gold_mapping[label_one] == seg2gold_mapping[label_two])
        #if not seg2gold_mapping[label_one] or not seg2gold_mapping[label_two]: ground_truth = False
        
        # get the small window around which to consider
        # TODO hardcoded change this
        sample_segment = segmentation[midpoint[IB_Z] - radii[IB_Z]:midpoint[IB_Z] + radii[IB_Z], midpoint[IB_Y] - radii[IB_Y]: midpoint[IB_Y] + radii[IB_Y], midpoint[IB_X] - radii[IB_X]:midpoint[IB_X] + radii[IB_X]]
        sample_gold = gold[midpoint[IB_Z] - radii[IB_Z]:midpoint[IB_Z] + radii[IB_Z], midpoint[IB_Y] - radii[IB_Y]: midpoint[IB_Y] + radii[IB_Y], midpoint[IB_X] - radii[IB_X]:midpoint[IB_X] + radii[IB_X]]

        seg2gold_sample = seg2gold.Mapping(sample_segment, sample_gold)
        ground_truth = (seg2gold_sample[label_one] == seg2gold_sample[label_two])
        
        # if either label iz zero there is no ground truth
        if not seg2gold_mapping[label_one] or not seg2gold_mapping[label_two]: 
            undetermined_candidates.append(SkeletonCandidate((label_one, label_two), midpoint, ground_truth))
            continue

        # create the candidate and add to the list
        candidate = SkeletonCandidate((label_one, label_two), midpoint, ground_truth)
        if ground_truth: positive_candidates.append(candidate)
        else: negative_candidates.append(candidate)

    return positive_candidates, negative_candidates, undetermined_candidates



# save the candidate files for the CNN
def SaveCandidates(output_filename, positive_candidates, negative_candidates, inference=False, undetermined_candidates=None):
    if not undetermined_candidates == None:
        candidates = undetermined_candidates
        random.shuffle(candidates)
    elif inference:
        # concatenate the two lists
        candidates = positive_candidates + negative_candidates
        random.shuffle(candidates)
    else:
        # randomly shuffle the arrays
        random.shuffle(positive_candidates)
        random.shuffle(negative_candidates)

        # get the minimum length of the two candidates - train in pairs
        npoints = max(len(positive_candidates), len(negative_candidates))
        positive_index = 0
        negative_index = 0

        # train in pairs, duplicate when needed
        candidates = []
        for _ in range(npoints):
            candidates.append(positive_candidates[positive_index])
            candidates.append(negative_candidates[negative_index])

            # increment the indices
            positive_index += 1
            negative_index += 1

            # handle dimension mismatch by reseting index and reshuffling array
            if positive_index >= len(positive_candidates): 
                positive_index = 0
                random.shuffle(positive_candidates)
            if negative_index >= len(negative_candidates): 
                negative_index = 0
                random.shuffle(negative_candidates)
            
    # write all candidates to the file
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('i', len(candidates)))

        # add every candidate to the binary file
        for candidate in candidates:
            # get the labels for this candidate
            label_one = candidate.labels[0]
            label_two = candidate.labels[1]

            # get the location of this candidate
            position = candidate.location

            # get the ground truth for this candidate
            ground_truth = candidate.ground_truth

            # write this candidate to the evaluation candidate list
            fd.write(struct.pack('QQQQQQ', label_one, label_two, position[IB_Z], position[IB_Y], position[IB_X], ground_truth))



def GenerateFeatures(prefix, threshold, maximum_distance):
    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)
    assert (segmentation.shape == gold.shape)

    # remove small connected components
    segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, threshold=threshold)

    # get the grid size and the world resolution
    grid_size = segmentation.shape
    world_res = dataIO.Resolution(prefix)

    # get the radius in grid coordinates
    ## TODO do not hardcode this!!
    radii = (20, 100, 100)



    # read in the skeletons, ignore the joints
    skeletons, _, endpoints = dataIO.ReadSkeletons(prefix, segmentation)

    # generate the kdtree
    locations, kdtree = GenerateKDTree(endpoints, world_res)

    # query the kdtree to find close neighbors within maximum distance
    neighbors = kdtree.query_ball_tree(kdtree, maximum_distance)

    # prune the neighbors
    neighbors = PruneNeighbors(neighbors, endpoints, radii, grid_size)



    # create a mapping from segmentation to gold
    seg2gold_mapping = seg2gold.Mapping(segmentation, gold)

    # generate all the candidates with the SkeletonFeature class
    positive_candidates, negative_candidates, undetermined_candidates = GenerateCandidates(neighbors, endpoints, segmentation, gold, seg2gold_mapping, radii)


    # print statistics
    print 'Results for {}, threshold {}, maximum distance {}:'.format(prefix, threshold, maximum_distance)
    print '  Positive examples: {}'.format(len(positive_candidates))
    print '  Negative examples: {}'.format(len(negative_candidates))
    print '  Ratio: {}'.format(len(negative_candidates) / float(len(positive_candidates)))

    # save the files
    train_filename = 'features/skeleton/{}-{}-{}nm-learning.candidates'.format(prefix, threshold, maximum_distance)
    forward_filename = 'features/skeleton/{}-{}-{}nm-inference.candidates'.format(prefix, threshold, maximum_distance)
    undetermined_filename = 'features/skeleton/{}-{}-{}nm-undetermined.candidates'.format(prefix, threshold, maximum_distance)

    SaveCandidates(train_filename, positive_candidates, negative_candidates, inference=False)
    SaveCandidates(forward_filename, positive_candidates, negative_candidates, inference=True)
    SaveCandidates(undetermined_filename, positive_candidates, negative_candidates, undetermined_candidates=undetermined_candidates)