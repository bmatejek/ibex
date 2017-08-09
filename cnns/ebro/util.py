import struct
import numpy as np
from numba import jit

from ibex.utilities.constants import *
from ibex.utilities import dataIO



# class that contains all import feature data
class EbroCandidate:
    def __init__(self, labels, location, ground_truth):
        self.labels = labels
        self.location = location
        self.ground_truth = ground_truth



# go from world coordinates to grid coordinates
def WorldToGrid(world_position, bounding_box):
    zdiff = world_position[IB_Z] - bounding_box.mins[IB_Z]
    ydiff = world_position[IB_Y] - bounding_box.mins[IB_Y]
    xdiff = world_position[IB_X] - bounding_box.mins[IB_X]

    return (zdiff, ydiff, xdiff)



# read in the gold file
def ReadGold(prefix_one, prefix_two, threshold, maximum_distance):
    # get the gold file
    gold_filename = 'gold/{}-{}-{}-{}nm.gold'.format(prefix_one, prefix_two, threshold, maximum_distance)

    # open the file and read all candidates
    with open(gold_filename, 'rb') as fd:
        ncandidates, = struct.unpack('i', fd.read(4))

        # read all of the ground truth
        ground_truth = []
        for iv in range(ncandidates):
            decision, = struct.unpack('i', fd.read(4))
            ground_truth.append(decision)

    # return generated ground truth
    return ground_truth



# read in the feature labels and locations
def ReadFeatures(prefix_one, prefix_two, threshold, maximum_distance):
    # get the feature filename
    feature_filename = 'features/ebro/{}-{}-{}-{}nm.candidates'.format(prefix_one, prefix_two, threshold, maximum_distance)

    # open the file and read candidates
    with open(feature_filename, 'rb') as fd:
        ncandidates, = struct.unpack('i', fd.read(4))

        # read all of the labels and locations
        labels = []
        locations = []
        for iv in range(ncandidates):
            label_one, label_two, centerz, centery, centerx, = struct.unpack('QQQQQ', fd.read(40))

            labels.append((label_one, label_two))
            locations.append((centerz, centery, centerx))

    # return the relevant information
    return labels, locations



# read in the counters
def ReadCounters(prefix_one, prefix_two, threshold, maximum_distance):
    # get the counter filename
    counter_filename = 'features/ebro/{}-{}-{}-{}nm.counts'.format(prefix_one, prefix_two, threshold, maximum_distance)

    # open the file and read counters
    with open(counter_filename, 'rb') as fd:
        ncandidates, = struct.unpack('i', fd.read(4))

        # read all of the counts
        counts_one = []
        counts_two = []
        overlap_counts = []
        for iv in range(ncandidates):
            count_one, count_two, overlap_count, = struct.unpack('QQQ', fd.read(24))

            counts_one.append(count_one)
            counts_two.append(count_two)
            overlap_counts.append(overlap_count)

    # return the relevant information
    return counts_one, counts_two, overlap_counts

    

# find the candidates for these prefixes, threshold and distance
def FindCandidates(prefix_one, prefix_two, threshold, maximum_distance, inference):
    # get the gold decisions for this arrangement
    ground_truth = ReadGold(prefix_one, prefix_two, threshold, maximum_distance)
    ncandidates = len(ground_truth)

    # get the features for this arrangment
    labels, locations = ReadFeatures(prefix_one, prefix_two, threshold, maximum_distance)

    positive_candidates = []
    negative_candidates = []
    candidates = []

    # only consider locations where there is legitimate ground truth
    for iv in range(ncandidates):
        # skip over undecided candidates
        if not inference and ground_truth[iv] == 2: continue

        candidate = EbroCandidate(labels[iv], locations[iv], ground_truth[iv])
        candidates.append(candidate)
        if candidate.ground_truth: positive_candidates.append(candidate)
        else: negative_candidates.append(candidate)

    if not inference:
        candidates = []
        for iv in range(min(len(positive_candidates), len(negative_candidates))):
            candidates.append(positive_candidates[iv])
            candidates.append(negative_candidates[iv])

    return candidates



@jit(nopython=True)
# is this position contained in the grid
def Contains(position, shape):
    if position[IB_Z] < 0 or position[IB_Z] >= shape[IB_Z]: return False
    if position[IB_Y] < 0 or position[IB_Y] >= shape[IB_Y]: return False
    if position[IB_X] < 0 or position[IB_X] >= shape[IB_X]: return False
    return True



@jit(nopython=True)
# scale the feature
def ScaleFeature(segmentations, images, labels, positions, width, radii):
    # constants for channels
    nchannels = width[3]
    assert (nchannels == 3 or nchannels == 4)
    GRID_ONE = 0
    GRID_TWO = 1
    EITHER_GRID = 2
    IMAGE_GRID = 3

    example = np.zeros((1, width[IB_Z], width[IB_Y], width[IB_X], nchannels), dtype=np.uint8)

    zres, yres, xres = segmentations[GRID_ONE].shape
    assert (segmentations[GRID_ONE].shape == segmentations[GRID_TWO].shape)

    # iterate over entire window
    for iz in range(width[IB_Z]):
        for iy in range(width[IB_Y]):
            for ix in range(width[IB_X]):
                # get the offset in the region of interest
                zoffset = int(iz * float(2 * radii[IB_Z]) / width[IB_Z] + 0.5)
                yoffset = int(iy * float(2 * radii[IB_Y]) / width[IB_Y] + 0.5)
                xoffset = int(ix * float(2 * radii[IB_X]) / width[IB_X] + 0.5)

                # get the locations for both segments
                location_one = (positions[GRID_ONE][IB_Z] + zoffset, positions[GRID_ONE][IB_Y] + yoffset, positions[GRID_ONE][IB_X] + xoffset)
                location_two = (positions[GRID_TWO][IB_Z] + zoffset, positions[GRID_TWO][IB_Y] + yoffset, positions[GRID_TWO][IB_X] + xoffset)

                # does the segment belong to the first grid
                if Contains(location_one, segmentations[GRID_ONE].shape):
                    if segmentations[GRID_ONE][location_one[IB_Z], location_one[IB_Y], location_one[IB_X]] == labels[GRID_ONE]:
                        example[0,iz,iy,ix,GRID_ONE] = 1
                        example[0,iz,iy,ix,EITHER_GRID] = 1
                    if (nchannels == 4): example[0,iz,iy,ix,IMAGE_GRID] = images[GRID_ONE][location_one[IB_Z], location_one[IB_Y], location_one[IB_X]]

                # does the segment belong to the second grid
                if Contains(location_two, segmentations[GRID_TWO].shape):
                    if segmentations[GRID_TWO][location_two[IB_Z], location_two[IB_Y], location_two[IB_X]] == labels[GRID_TWO]:
                        example[0,iz,iy,ix,GRID_TWO] = 1
                        example[0,iz,iy,ix,EITHER_GRID] = 1
                    if (nchannels == 4): example[0,iz,iy,ix,IMAGE_GRID] = images[GRID_TWO][location_two[IB_Z], location_two[IB_Y], location_two[IB_X]]

    return example



# extract this candidate for the given segmentations
def ExtractFeature(segmentations, images, bboxes, candidate, width, radii, rotation):
    # only 16 unique rotations for anisotropic data sets
    assert (rotation < 16)
    GRID_ONE = 0
    GRID_TWO = 1

    # put the data in convenient forms
    labels = candidate.labels
    location = candidate.location

    # update the location from the smallest corner of region of interest
    location = (location[IB_Z] - radii[IB_Z], location[IB_Y] - radii[IB_Y], location[IB_X] - radii[IB_X])

    # get the grid locations
    positions = (WorldToGrid(location, bboxes[GRID_ONE]), WorldToGrid(location, bboxes[GRID_TWO]))
    example = ScaleFeature(segmentations, images, labels, positions, width, radii)

    # flip x axis? -> add 1 because of extra filler channel
    if rotation % 2: example = np.flip(example, IB_X + 1)
    # flip z axis?
    if (rotation / 2) % 2: example = np.flip(example, IB_Z + 1)

    # rotate in y?
    yrotation = rotation / 4
    example = np.rot90(example, k=yrotation, axes=(IB_X + 1, IB_Y + 1))

    return example



# save all of the features for these prefixes
def SaveFeatures(prefix_one, prefix_two, threshold, maximum_distance):
    # read in both segmentation and image files
    segmentations = (dataIO.ReadSegmentationData(prefix_one), dataIO.ReadSegmentationData(prefix_two))
    assert (segmentations[0].shape == segmentations[1].shape)
    images = (dataIO.ReadImageData(prefix_one), dataIO.ReadImageData(prefix_two))
    assert (images[0].shape == images[1].shape)
    bboxes = (dataIO.GetWorldBBox(prefix_one), dataIO.GetWorldBBox(prefix_two))
    world_res = dataIO.Resolution(prefix_one)
    assert (world_res == dataIO.Resolution(prefix_two))

    # get the radii for this feature
    radii = (maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X])
    width = (2 * radii[IB_Z], 2 * radii[IB_Y], 2 * radii[IB_X], 3)

    # get all of the candidates for these prefixes
    candidates = FindCandidates(prefix_one, prefix_two, threshold, maximum_distance, True)
    ncandidates = len(candidates)

    # iterate over all candidates
    for iv, candidate in enumerate(candidates):
        # get the example with zero rtation
        example = ExtractFeature(segmentations, images, bboxes, candidate, width, radii, 0)

        # compress the channels
        compressed_output = np.zeros((width[IB_Z], width[IB_Y], width[IB_X]), dtype=np.uint8)
        compressed_output[example[0,:,:,:,0] == 1] = 1
        compressed_output[example[0,:,:,:,1] == 1] = 2
        # both candidates are present at this location
        compressed_output[np.logical_and(example[0,:,:,:,0] == 1, example[0,:,:,:,1] == 1)] = 3

        # save the output file
        filename = 'features/ebro/{}-{}/{}-{}nm-{:05d}.h5'.format(prefix_one, prefix_two, threshold, maximum_distance, iv)
        dataIO.WriteH5File(compressed_output, filename, 'main')