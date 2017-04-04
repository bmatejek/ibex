import numpy as np
import struct
import random
from scipy.spatial import KDTree
from ibex.utilities import dataIO
from ibex.transforms import seg2gold
from util import Candidate



# generate a kdtree from the skeleton endpoints
def GenerateKDTree(endpoints, world_res):
    # get the number of endpoints
    npoints = len(endpoints)

    # create an array for locations
    locations = np.zeros((npoints, 3), dtype=np.float32)

    # go through every endpoint
    for ip, endpoint in enumerate(endpoints):
        locations[ip,:] = endpoint.WorldPoint(world_res)

    kdtree = KDTree(locations)

    return locations, kdtree



# further restrict the locations for candidates
def PruneNeighbors(neighbors, endpoints, radii, max_distance, grid_size):
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
            for dim in range(3):
                if (midpoint[dim] - radii[dim] < 0): interior_neighbor = False
                if (midpoint[dim] + radii[dim] > grid_size[dim]): interior_neighbor = False
            if not interior_neighbor: continue

            # append these neighbors to the 
            pruned_neighbors.append((neighbor_one, neighbor_two))

    return pruned_neighbors



# create the skeleton merge candidate
def GenerateCandidates(neighbors, endpoints, seg2gold):
    candidates = []

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
        candidates.append(candidate)

    return candidates



# save the candidate files for the CNN
def SaveCandidates(prefix, maximum_distance, candidates, radii, forward=False):
    # randomly shuffle the array
    random.shuffle(candidates)

    # count the number of positive and negative candidates
    npositive_candidates = 0
    nnegative_candidates = 0

    for candidate in candidates:
        if candidate.GroundTruth():
            npositive_candidates += 1
        else:
            nnegative_candidates += 1

    # just checking...if this ever breaks cap the number of negative candidates
    assert (npositive_candidates < nnegative_candidates)

    # if this is for training, make the number of positive and negative examples equal
    if forward: pruned_candidates = candidates
    else:
        positive_candidates = []
        negative_candidates = []

        nnegative_examples = 0

        for candidate in candidates:
            if candidate.GroundTruth():
                positive_candidates.append(candidate)
            else:
                # this make sures that there
                if (nnegative_examples == npositive_candidates): continue
                else:
                    negative_candidates.append(candidate)

                    # increment the number of negative candidates seen
                    nnegative_examples += 1

        # create an array of pruned examples
        pruned_candidates = []
        for iv in range(len(positive_candidates)):
            pruned_candidates.append(positive_candidates[iv])
            pruned_candidates.append(negative_candidates[iv])

        
    # get the output filename
    if forward: output_filename = 'skeletons/{0}_{1}nm_forward.candidates'.format(prefix, maximum_distance)
    else: output_filename = 'skeletons/{0}_{1}nm_train.candidates'.format(prefix, maximum_distance)

    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('I', len(pruned_candidates)))

        # add every candidate to the binary file
        for candidate in pruned_candidates:
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
def GenerateFeatures(prefix, maximum_distance, verbose=1):
    # read the segmentation and gold datasets
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)
    assert (segmentation.shape == gold.shape)

    # get the grid size and the world resolution in (z, y, x)
    grid_size = segmentation.shape
    world_res = dataIO.ReadMetaData(prefix)

    # print critical information about candidate extraction
    if verbose:
        print 'Generating candidates for ' + prefix + ':'
        print '  Considering neighboring segments within a {:d}nm radius.'.format(maximum_distance)
        # print the grid size (x, y, z)
        print '  Grid Size: {:d} {:d} {:d}'.format(grid_size[2], grid_size[1], grid_size[0])
        # print the sampling resolution (x, y, z)
        print '  Sampling Resolution: {:d}nm x {:d}nm x {:d}nm'.format(world_res[2], world_res[1], world_res[0])

    # read in the skeletons (ignore the joints here)
    skeletons, joints, endpoints = dataIO.ReadSkeletons(prefix, segmentation)        

    # generate the kdtree
    locations, kdtree = GenerateKDTree(endpoints, world_res)

    # query the kdtree to find close neighbors
    neighbors = kdtree.query_ball_tree(kdtree, maximum_distance)

    # get the radius in grid coordinates
    radii = (maximum_distance / world_res[0], maximum_distance / world_res[1], maximum_distance / world_res[2])

    # find all locations where potential merges should occur
    neighbors = PruneNeighbors(neighbors, endpoints, radii, maximum_distance, grid_size)

    # create a mapping from segmentation to gold
    seg2gold_mapping = seg2gold.Mapping(segmentation, gold)

    # generate all of the candidates with the SkeletonFeature class
    candidates = GenerateCandidates(neighbors, endpoints, seg2gold_mapping)

    # save the skeleton candidates
    SaveCandidates(prefix, maximum_distance, candidates, radii, forward=True)
    SaveCandidates(prefix, maximum_distance, candidates, radii, forward=False)