import numpy as np
import struct
import random
from scipy.spatial import KDTree
from ibex.utilities import dataIO
from ibex.utilities.constants import *
from ibex.transforms import seg2gold, seg2seg
from util import Candidate



# generate a kdtree from the skeleton endpoints
def GenerateKDTree(endpoints, world_res):
    # get the number of endpoints
    npoints = len(endpoints)

    # create an array for locations
    locations = np.zeros((npoints, NDIMS), dtype=np.float32)

    # go through every endpoint
    for ip, endpoint in enumerate(endpoints):
        locations[ip,:] = endpoint.WorldPoint(world_res)

    kdtree = KDTree(locations)

    return locations, kdtree



# further restrict the locations for candidates
def PruneNeighbors(neighbors, endpoints, radii, grid_size):
    # create an array for merge locations
    pruned_neighbors = []

    # consider all pairs of neighbors
    for neighbor_one, neighbor_list in enumerate(neighbors):
        for neighbor_two in neighbor_list:
            # to avoid double counting neighbors
            if (neighbor_one > neighbor_two): continue

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

            # if either x or y is too close to boundary skip
            # this allows for 4 translations in training
            if (midpoint[IB_X] - radii[IB_X] < 0): interior_neighbor = False
            if (midpoint[IB_Y] - radii[IB_Y] < 0): interior_neighbor = False
            if (midpoint[IB_X] + radii[IB_X] >= grid_size[IB_X]): interior_neighbor = False
            if (midpoint[IB_Y] + radii[IB_Y] >= grid_size[IB_Y]): interior_neighbor = False

            if not interior_neighbor: continue

            # append these neighbors to the 
            pruned_neighbors.append((neighbor_one, neighbor_two))

    return pruned_neighbors



# create the skeleton merge candidate
def GenerateCandidates(neighbors, endpoints, seg2gold):
    positive_candidates = []
    negative_candidates = []

    # iterate through all of the neighbors
    for (neighbor_one, neighbor_two) in neighbors:
        # get the label for these neighbors
        label_one = endpoints[neighbor_one].Label()
        label_two = endpoints[neighbor_two].Label()

        # get the location of both endpoints
        point_one = endpoints[neighbor_one].GridPoint()
        point_two = endpoints[neighbor_two].GridPoint()

        # get the midpoint between the endpoints
        midpoint = (point_one + point_two) / 2

        # should these neighbors merge?
        ground_truth = (seg2gold[label_one] == seg2gold[label_two])

        # create the candidate and add to the list
        candidate = Candidate(label_one, label_two, midpoint, ground_truth)
        if ground_truth:
            positive_candidates.append(candidate)
        else:
            negative_candidates.append(candidate)

    return positive_candidates, negative_candidates



# save the candidate files for the CNN
def SaveCandidates(output_filename, positive_candidates, negative_candidates, forward=False):
    if forward:
        # concatenate the two lists
        candidates = positive_candidates + negative_candidates
        random.shuffle(candidates)
    else:
        # randomly shuffle the arrays
        random.shuffle(positive_candidates)
        random.shuffle(negative_candidates)

        # get the minimum length of the two candidates - train in pairs
        min_length = min(len(positive_candidates), len(negative_candidates))
        
        # create an array of positive + negative candidate pairs
        candidates = []
        for index in range(min_length):
            candidates.append(positive_candidates[index])
            candidates.append(negative_candidates[index])
            
    # write all candidates to the file
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('I', len(candidates)))

        # add every candidate to the binary file
        for candidate in candidates:
            # get the labels for this candidate
            label_one = candidate.LabelOne()
            label_two = candidate.LabelTwo()

            # get the location of this candidate
            position = candidate.Location()

            # get the ground truth for this candidate
            ground_truth = candidate.GroundTruth()

            # write this candidate to the evaluation candidate list
            fd.write(struct.pack('QQQQQQ', label_one, label_two, position[0], position[1], position[2], ground_truth))



# generate the candidates for a given segmentation
def GenerateFeatures(prefix, maximum_distance, threshold=10000, verbose=1):
    # read the segmentation and gold datasets
    segmentation = dataIO.ReadSegmentationData(prefix).astype(dtype=np.uint64)
    print '1: {}'.format(segmentation.dtype)
    gold = dataIO.ReadGoldData(prefix).astype(dtype=np.uint64)
    assert (segmentation.shape == gold.shape)

    # remove all components under the threshold size
    print '2: {}'.format(len(np.unique(segmentation)))
    print '3: {}'.format(segmentation.shape)
    print '4: {}'.format(segmentation)
    segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, min_size=threshold)
    print '9: {}'.format(segmentation)

    print '10: {}'.format(segmentation.shape)
    print '11: {}'.format(segmentation.dtype)
    print '12: {}'.format(np.amax(segmentation))
    print '13: {}'.format(segmentation[0,0,0])
    print '14: {}'.format(len(np.unique(segmentation)))
    print '15: {}'.format(segmentation.shape)
    
    # get the grid size and the world resolution in (z, y, x)
    grid_size = segmentation.shape
    world_res = dataIO.ReadMetaData(prefix)
    
    # print critical information about candidate extraction
    if verbose:
        print 'Generating candidates for ' + prefix + ':'
        print '  Considering neighboring segments within a {:d}nm radius.'.format(maximum_distance)
        # print the grid size (x, y, z)
        print '  Grid Size: {:d} {:d} {:d}'.format(grid_size[IB_Z], grid_size[IB_Y], grid_size[IB_X])
        # print the sampling resolution (x, y, z)
        print '  Sampling Resolution: {:d}nm x {:d}nm x {:d}nm'.format(world_res[IB_Z], world_res[IB_Y], world_res[IB_X])

    # read in the skeletons (ignore the joints here)
    skeletons, joints, endpoints = dataIO.ReadSkeletons(prefix, segmentation)        
    print 'Y'
    # generate the kdtree
    locations, kdtree = GenerateKDTree(endpoints, world_res)
    print 'Z'
    # query the kdtree to find close neighbors
    neighbors = kdtree.query_ball_tree(kdtree, maximum_distance)
    print 'W'
    # get the radius in grid coordinates
    radii = (maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X])
    print 'A'
    # find all locations where potential merges should occur
    neighbors = PruneNeighbors(neighbors, endpoints, radii, grid_size)
    print 'B'
    # create a mapping from segmentation to gold
    seg2gold_mapping = seg2gold.Mapping(segmentation, gold)
    print 'C'
    # generate all of the candidates with the SkeletonFeature class
    positive_candidates, negative_candidates = GenerateCandidates(neighbors, endpoints, seg2gold_mapping)
    print 'D'
    # print the number of candidates found
    if verbose:
        print 'Found candidates:'
        print '  {} positive'.format(len(positive_candidates))
        print '  {} negative'.format(len(negative_candidates))
    print 'E'
    # get the output filename
    forward_filename = 'skeletons/candidates/{}-{}nm_forward.candidates'.format(prefix, maximum_distance)
    train_filename = 'skeletons/candidates/{}-{}nm_train.candidates'.format(prefix, maximum_distance)
    print 'F'
    SaveCandidates(forward_filename, positive_candidates, negative_candidates, forward=True)
    SaveCandidates(train_filename, positive_candidates, negative_candidates, forward=False)
    print 'G'
