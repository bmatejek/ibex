import math
import time
import numpy as np
import random
import struct
from numba import jit

from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.transforms import seg2seg, seg2gold
from ibex.cnns.skeleton.util import SkeletonCandidate
from ibex.data_structures import unionfind
from PixelPred2Seg import comparestacks



# save the candidate files for the CNN
def SaveCandidates(output_filename, positive_candidates, negative_candidates, inference=False, validation=False, undetermined_candidates=None):
    if not undetermined_candidates == None:
        candidates = undetermined_candidates
        random.shuffle(candidates)
    elif inference:
        # concatenate the two lists
        candidates = positive_candidates + negative_candidates
        random.shuffle(candidates)
    else:
        positive_threshold = int(math.floor(0.80 * len(positive_candidates)))
        negative_threshold = int(math.floor(0.80 * len(negative_candidates)))
        
        if not validation:
            positive_candidates = positive_candidates[:positive_threshold]
            negative_candidates = negative_candidates[:negative_threshold]
        else:
            positive_candidates = positive_candidates[positive_threshold:]
            negative_candidates = negative_candidates[negative_threshold:]

        # shuffle the positive and negative candidates
        random.shuffle(positive_candidates)
        random.shuffle(negative_candidates)
            
        # get the maximum length of the two candidates - train in pairs
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


@jit(nopython=True)
def FindNeighboringCandidates(segmentation, centroid, candidates, maximum_distance, network_distance, world_res):
    # useful variables
    zres, yres, xres = segmentation.shape
    max_label = np.amax(segmentation) + 1

    # get the radii and label for this centroid
    radii = np.int64((maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X]))
    network_radii = np.int64((network_distance / world_res[IB_Z], network_distance / world_res[IB_Y], network_distance / world_res[IB_X]))
    label = segmentation[centroid[IB_Z],centroid[IB_Y],centroid[IB_X]]

    # iterate through all the pixels close to the centroid
    for iz in range(centroid[IB_Z]-radii[IB_Z], centroid[IB_Z]+radii[IB_Z]+1):
        if iz < 0 or iz > zres - 1: continue
        for iy in range(centroid[IB_Y]-radii[IB_Y], centroid[IB_Y]+radii[IB_Y]+1):
            if iy < 0 or iy > yres - 1: continue
            for ix in range(centroid[IB_X]-radii[IB_X], centroid[IB_X]+radii[IB_X]+1):
                if ix < 0 or ix > xres - 1: continue
                # skip extracellular and locations with the same label
                if not segmentation[iz,iy,ix]: continue
                if segmentation[iz,iy,ix] == label: continue

                # get the distance from the centroid
                distance = math.sqrt((world_res[IB_Z] * (centroid[IB_Z] - iz)) * (world_res[IB_Z] * (centroid[IB_Z] - iz))  + (world_res[IB_Y] * (centroid[IB_Y] - iy)) * (world_res[IB_Y] * (centroid[IB_Y] - iy)) + (world_res[IB_X] * (centroid[IB_X] - ix)) * (world_res[IB_X] * (centroid[IB_X] - ix)))
                if distance > maximum_distance: continue

                # is there already a closer location
                neighbor_label = segmentation[iz,iy,ix]
                candidates.add(neighbor_label)

@jit(nopython=True)
def Baseline(segmentation, baseline_candidates):
    # find all candidates that have adjacent voxels
    zres, yres, xres = segmentation.shape

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if not segmentation[iz,iy,ix]: continue
                label = segmentation[iz,iy,ix]

                # get the 6-connected neighbors to this segment
                if iz and segmentation[iz-1,iy,ix]:
                    baseline_candidates.add((label, segmentation[iz-1,iy,ix]))
                    baseline_candidates.add((segmentation[iz-1,iy,ix], label))
                if iy and segmentation[iz,iy-1,ix]:
                    baseline_candidates.add((label, segmentation[iz,iy-1,ix]))
                    baseline_candidates.add((segmentation[iz,iy-1,ix], label))
                if ix and segmentation[iz,iy,ix-1]:
                    baseline_candidates.add((label, segmentation[iz,iy,ix-1]))
                    baseline_candidates.add((segmentation[iz,iy,ix-1], label))

# generate features for this prefix
def GenerateFeatures(prefix, threshold, maximum_distance, network_distance, endpoint_distance, training_data):
    # read in the relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)
    assert (segmentation.shape == gold.shape)
    zres, yres, xres = segmentation.shape

    # get the mapping from segmentation to gold
    seg2gold_mapping = seg2gold.Mapping(segmentation, gold, low_threshold=0.10, high_threshold=0.80)

    # remove small connceted components
    segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, threshold=threshold).astype(np.int64)
    max_label = np.amax(segmentation) + 1

    # get the set of baseline candidates
    baseline_candidates = set()
    baseline_candidates.add((np.int64(0), np.int64(0)))

    Baseline(segmentation, baseline_candidates)

    baseline_positive = []
    baseline_negative = []

    for candidate in baseline_candidates:
        label_one = candidate[0]
        label_two = candidate[1]

        if not label_one or not label_two: continue

        # only consider (label_one, label_two) where label_two is larger
        if (label_two <= label_one): continue
        if not seg2gold_mapping[label_one] or not seg2gold_mapping[label_two]: continue

        ground_truth = (seg2gold_mapping[label_one] == seg2gold_mapping[label_two])

        if ground_truth: baseline_positive.append(SkeletonCandidate((label_one, label_two), (0, 0, 0), ground_truth))
        else: baseline_negative.append(SkeletonCandidate((label_one, label_two), (0, 0, 0), ground_truth))

    print 'Baseline:'
    print '  Positive Candidates: {}'.format(len(baseline_positive))
    print '  Negative Candidates: {}'.format(len(baseline_negative))
    print '  Ratio: {}'.format(100 * float(len(baseline_positive)) / (len(baseline_positive) + len(baseline_negative)))

    # get the grid size and the world resolution
    grid_size = segmentation.shape
    world_res = dataIO.Resolution(prefix)

    # get the radius in grid coordinates
    radii = np.int64((maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X]))
    network_radii = np.int64((network_distance / world_res[IB_Z], network_distance / world_res[IB_Y], network_distance / world_res[IB_X]))
    
    # get all of the skeletons
    skeletons, _, endpoints = dataIO.ReadSkeletons(prefix, segmentation)


    # get the set of all pairs considered
    endpoint_candidates = [set() for _ in range(len(endpoints))]

    # iterate through all skeleton endpoints
    for ie, endpoint in enumerate(endpoints):
        # extract the region around this endpoint
        label = endpoint.label
        centroid = endpoint.GridPoint()

        # find the candidates from this endpoint
        candidates = set()
        candidates.add(0)
        FindNeighboringCandidates(segmentation, centroid, candidates, maximum_distance, network_distance, world_res)

        for candidate in candidates:
            if not candidate: continue
            endpoint_candidates[ie].add(candidate)


    threshold_low_candidates = set()
    # go through all the endpoint candidates
    for ie, endpoint in enumerate(endpoints):
        label = endpoint.label
        for neighbor_label in endpoint_candidates[ie]:
            threshold_low_candidates.add((label, neighbor_label))
            threshold_low_candidates.add((neighbor_label, label))

    threshold_low_positive = []
    threshold_low_negative = []

    for candidate in threshold_low_candidates:
        label_one = candidate[0]
        label_two = candidate[1]

        if not label_one or not label_two: continue

        # only consider (label_one, label_two) where label two is higher
        if (label_two <= label_one): continue
        if not seg2gold_mapping[label_one] or not seg2gold_mapping[label_two]: continue

        ground_truth = (seg2gold_mapping[label_one] == seg2gold_mapping[label_two])

        if ground_truth: threshold_low_positive.append(SkeletonCandidate((label_one, label_two), (0, 0, 0), ground_truth))
        else: threshold_low_negative.append(SkeletonCandidate((label_one, label_two), (0, 0, 0), ground_truth))

    print 'Threshold Low:'
    print '  Positive Candidates: {}'.format(len(threshold_low_positive))
    print '  Negative Candidates: {}'.format(len(threshold_low_negative))
    print '  Ratio: {}'.format(100 * float(len(threshold_low_positive)) / (len(threshold_low_positive) + len(threshold_low_negative)))

    ##################################
    #### BEGIN PRUNING CANDIDATES ####
    ##################################
        
    endpoint_pairs = set()

    # iterate through all skeleton endpoints
    for ie, endpoint in enumerate(endpoints):
        # get the endpoint location
        label = endpoint.label

        # find the candidates from this endpoint after pruning
        candidates = set()

        # go through all currently considered endpoints
        for neighbor_label in endpoint_candidates[ie]:
            for neighbor_endpoint in skeletons[neighbor_label].endpoints:
                # get the distance
                deltas = endpoint.WorldPoint(world_res) - neighbor_endpoint.WorldPoint(world_res)
                distance = math.sqrt(deltas[IB_Z] * deltas[IB_Z] + deltas[IB_Y] * deltas[IB_Y] + deltas[IB_X] * deltas[IB_X])

                if distance < endpoint_distance:
                    candidates.add(neighbor_label)
                    endpoint_pairs.add((endpoint, neighbor_endpoint))


    # find the smallest pair between endpoints
    smallest_distances = (endpoint_distance + 1) * np.ones((max_label, max_label), dtype=np.float32)
    midpoints = np.zeros((max_label, max_label, 3), dtype=np.uint32)
    
    # go through all the endpoint pairs and prune out the ones with bad angles
    for endpoint_pair in endpoint_pairs:
        # get the endpoints under consideration
        endpoint_one = endpoint_pair[0]
        endpoint_two = endpoint_pair[1]
        label_one = endpoint_one.label
        label_two = endpoint_two.label

        midpoint = (endpoint_one.GridPoint() + endpoint_two.GridPoint()) / 2

        # make sure the bounding box fits
        # valid_location = True
        # for dim in range(NDIMS):
        #     if midpoint[dim]-network_radii[dim] < 0: valid_location = False
        #     if midpoint[dim]+network_radii[dim] > grid_size[dim]: valid_location = False
        # if not valid_location: continue

        # get the distance between the endpoints
        deltas = endpoint_one.WorldPoint(world_res) - endpoint_two.WorldPoint(world_res)
        distance = math.sqrt(deltas[IB_Z] * deltas[IB_Z] + deltas[IB_Y] * deltas[IB_Y] + deltas[IB_X] * deltas[IB_X])
        if distance > smallest_distances[label_one,label_two]: continue

        smallest_distances[label_one,label_two] = distance
        smallest_distances[label_two,label_one] = distance
        midpoints[label_one,label_two,:] = midpoint
        midpoints[label_two,label_one,:] = midpoint


    # create list of candidates
    positive_candidates = []
    negative_candidates = []
    undetermined_candidates = []

    for label_one in range(0, max_label):
        for label_two in range(label_one + 1, max_label):
            if smallest_distances[label_one,label_two] > endpoint_distance: continue
            ground_truth = (seg2gold_mapping[label_one] == seg2gold_mapping[label_two])
            candidate = SkeletonCandidate((label_one, label_two), midpoints[label_one,label_two,:], ground_truth)

            if not seg2gold_mapping[label_one] or not seg2gold_mapping[label_two]: undetermined_candidates.append(candidate)
            elif ground_truth: positive_candidates.append(candidate)
            else: negative_candidates.append(candidate)

    # # save the files
    # train_filename = 'features/skeleton/{}-{}-{}nm-{}nm-training.candidates'.format(prefix, threshold, maximum_distance, network_distance)
    # validation_filename = 'features/skeleton/{}-{}-{}nm-{}nm-validation.candidates'.format(prefix, threshold, maximum_distance, network_distance)
    # forward_filename = 'features/skeleton/{}-{}-{}nm-{}nm-inference.candidates'.format(prefix, threshold, maximum_distance, network_distance)
    # undetermined_filename = 'features/skeleton/{}-{}-{}nm-{}nm-undetermined.candidates'.format(prefix, threshold, maximum_distance, network_distance)

    # if training_data:
    #     SaveCandidates(train_filename, positive_candidates, negative_candidates, inference=False, validation=False)
    #     SaveCandidates(validation_filename, positive_candidates, negative_candidates, inference=False, validation=True)
    # SaveCandidates(forward_filename, positive_candidates, negative_candidates, inference=True)
    # SaveCandidates(undetermined_filename, positive_candidates, negative_candidates, undetermined_candidates=undetermined_candidates)

    print 'Threshold High:'
    print '  Positive Candidates: {}'.format(len(positive_candidates))
    print '  Negative Candidates: {}'.format(len(negative_candidates))
    # print '  Undetermined Candidates: {}'.format(len(undetermined_candidates))
    print '  Ratio: {}'.format(100 * len(positive_candidates) / float(len(positive_candidates) + len(negative_candidates)))

    misses = []
    catches = []

    # get the list of all candidates the baseline found that we did not
    for their_candidate in baseline_positive:
        their_labels = their_candidate.labels

        found = False
        for our_candidate in positive_candidates:
            our_labels = our_candidate.labels

            if our_labels[0] == their_labels[0] and our_labels[1] == their_labels[1]:
                found = True
                break

        if not found:
            misses.append(their_labels)

    for our_candidate in positive_candidates:
        our_labels = our_candidate.labels

        found = False
        for their_candidate in baseline_positive:
            their_labels = their_candidate.labels

            if our_labels[0] == their_labels[0] and our_labels[1] == their_labels[1]:
                found = True
                break

        if not found:
            catches.append(our_labels)

    print 'Number Missed: {}'.format(len(misses))
    print 'Number Caught: {}'.format(len(catches))
    print '\n'

    with open('cvpr/{}-skeleton-misses.txt'.format(prefix), 'wb') as fd:
        fd.write(struct.pack('i', len(misses)))

        for miss in misses:
            fd.write(struct.pack('ii', miss[0], miss[1]))
    with open('cvpr/{}-skeleton-catches.txt'.format(prefix), 'wb') as fd:
        fd.write(struct.pack('i', len(catches)))

        for catch in catches:
            fd.write(struct.pack('ii', catch[0], catch[1]))

#    # perform some tests to see how well this method can do
#    max_value = np.uint64(np.amax(segmentation) + 1)
#    union_find = [unionfind.UnionFindElement(iv) for iv in range(max_value)]

#    # iterate over all collapsed edges
#    for candidate in positive_candidates:
#        label_one, label_two = candidate.labels
#        unionfind.Union(union_find[label_one], union_find[label_two])

#    # create a mapping for the labels
#    mapping = np.zeros(max_value, dtype=np.uint64)
#    for iv in range(max_value):
#        mapping[iv] = unionfind.Find(union_find[iv]).label
#    opt_segmentation = seg2seg.MapLabels(segmentation, mapping)
#    comparestacks.Evaluate(opt_segmentation, gold)
