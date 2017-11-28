import numpy as np
import random
import struct
import math

from ibex.cnns.nuclear.util import NuclearCandidate
from ibex.utilities import dataIO
from ibex.utilities.constants import *
from ibex.transforms import seg2seg, seg2gold



# save the candidate files for the CNN
def SaveCandidates(output_filename, positive_candidates, negative_candidates, inference=False, validation=False):
    if inference:
        candidates = positive_candidates + negative_candidates
        random.shuffle(candidates)
    else:
        # randomly shuffle arrays
        random.seed(0)
        random.shuffle(positive_candidates)
        random.shuffle(negative_candidates)

        positive_threshold = int(math.floor(0.80 * len(positive_candidates)))
        negative_threshold = int(math.floor(0.80 * len(negative_candidates)))

        if not validation:
            positive_candidates = positive_candidates[:positive_threshold]
            negative_candidates = negative_candidates[:negative_threshold]
        else:
            positive_candidates = positive_candidates[positive_threshold:]
            negative_candidates = negative_candidates[negative_threshold:]

        # get the maximum length of the 
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

    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('i', len(candidates)))

        # add every candidate to the binary file
        for candidate in candidates:
            # get the candidate label
            label = candidate.label

            # get the potential split location
            position = candidate.location

            # get the ground truth for this candidate
            ground_truth = candidate.ground_truth

            fd.write(struct.pack('QQQQQ', label, position[IB_Z], position[IB_Y], position[IB_X], ground_truth))



# generate features for this prefix
def GenerateFeatures(prefix, threshold, network_distance):
    # read in the relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)
    assert (segmentation.shape == gold.shape)
    zres, yres, xres = segmentation.shape

    # get the mapping from the segmentation to gold
    seg2gold_mapping = seg2gold.Mapping(segmentation, gold, low_threshold=0.10, high_threshold=0.80)

    # remove small connected components
    segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, threshold=threshold).astype(np.int64)
    max_label = np.amax(segmentation) + 1

    # get the grid size and the world resolution
    grid_size = segmentation.shape
    world_res = dataIO.Resolution(prefix)

    # get the radius in grid coordinates
    network_radii = np.int64((network_distance / world_res[IB_Z], network_distance / world_res[IB_Y], network_distance / world_res[IB_X]))


    # get all of the skeletons
    skeletons, _, _ = dataIO.ReadSkeletons(prefix, segmentation)

    npositive_instances = [0 for _ in range(10)]
    nnegative_instances = [0 for _ in range(10)]

    positive_candidates = []
    negative_candidates = []

    # iterate over all skeletons
    for skeleton in skeletons:
        label = skeleton.label
        joints = skeleton.joints

        # iterate over all joints
        for joint in joints:
            # get the gold value at this location
            location = joint.GridPoint()
            gold_label = gold[location[IB_Z], location[IB_Y], location[IB_X]]            


            # make sure the bounding box fits
            valid_location = True
            for dim in range(NDIMS):
                if location[dim]-network_radii[dim] < 0: valid_location = False
                if location[dim]+network_radii[dim] > grid_size[dim]: valid_location = False
            if not valid_location: continue


            if not gold_label: continue

            neighbors = joint.Neighbors()
            should_split = False

            if len(neighbors) <= 2: continue

            # get the gold for every neighbor
            for neighbor in neighbors:
                neighbor_location = neighbor.GridPoint()
                neighbor_gold_label = gold[neighbor_location[IB_Z], neighbor_location[IB_Y], neighbor_location[IB_X]]

                # get the gold value here

                if not gold_label == neighbor_gold_label and gold_label and neighbor_gold_label: should_split = True

            if should_split: npositive_instances[len(neighbors)] += 1
            else: nnegative_instances[len(neighbors)] += 1

            candidate = NuclearCandidate(label, location, should_split)
            if should_split: positive_candidates.append(candidate)
            else: negative_candidates.append(candidate)


    train_filename = 'features/nuclear/{}-{}-{}nm-training.candidates'.format(prefix, threshold, network_distance)
    validation_filename = 'features/nuclear/{}-{}-{}nm-validation.candidates'.format(prefix, threshold, network_distance)
    forward_filename = 'features/nuclear/{}-{}-{}nm-inference.candidates'.format(prefix, threshold, network_distance)
    SaveCandidates(train_filename, positive_candidates, negative_candidates, inference=False, validation=False)
    SaveCandidates(validation_filename, positive_candidates, negative_candidates, inference=False, validation=True)
    SaveCandidates(forward_filename, positive_candidates, negative_candidates, inference=True)

    print '  Positive Candidates: {}'.format(len(positive_candidates))
    print '  Negative Candidates: {}'.format(len(negative_candidates))
    print '  Ratio: {}'.format(len(negative_candidates) / float(len(positive_candidates)))