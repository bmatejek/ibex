import struct
from ibex.utilities.constants import *
from numba import jit
import numpy as np
from ibex.utilities import dataIO



# go from world coordinates to grid coordinates
def WorldToGrid(world_position, bounding_box):
    zdiff = world_position[IB_Z] - bounding_box.Min(IB_Z)
    ydiff = world_position[IB_Y] - bounding_box.Min(IB_Y)
    xdiff = world_position[IB_X] - bounding_box.Min(IB_X)

    return (zdiff, ydiff, xdiff)



# read in all of the counters
def ReadCounters(prefix_one, prefix_two, threshold, maximum_distance):
    # get the counter filename
    counter_filename = 'features/ebro/{}-{}-{}-{}nm.counts'.format(prefix_one, prefix_two, threshold, maximum_distance)

    # open the file and read candidates
    with open(counter_filename, 'rb') as fd:
        ncandidates, = struct.unpack('Q', fd.read(8))

        # read all of the various counting variables
        label_one_counts = []
        label_two_counts = []
        overlap_counts = []
        scores = []
        for iv in range(ncandidates):
            label_one_count, label_two_count, overlap_count, score, = struct.unpack('QQQd', fd.read(32))

            label_one_counts.append(label_one_count)
            label_two_counts.append(label_two_count)
            overlap_counts.append(overlap_count)
            scores.append(score)

    # return the relevant information
    return label_one_counts, label_two_counts, overlap_counts, scores



# read in the feature labels and locations
def ReadFeatures(prefix_one, prefix_two, threshold, maximum_distance):
    # get the feature filename
    feature_filename = 'features/ebro/{}-{}-{}-{}nm.candidates'.format(prefix_one, prefix_two, threshold, maximum_distance)

    # open the file and read candidates
    with open(feature_filename, 'rb') as fd:
        ncandidates, = struct.unpack('Q', fd.read(8))

        # read all of the labels and locations
        labels = []
        locations = []
        for iv in range(ncandidates):
            label_one, label_two, centerx, centery, centerz, = struct.unpack('QQQQQ', fd.read(40))

            labels.append((label_one, label_two))
            locations.append((centerz, centery, centerx))

    # return the relevant information
    return labels, locations



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



# class that contains all import feature data
class Candidate:
    def __init__(self, labels, location, ground_truth):
        self.labels = labels
        self.location = location
        self.ground_truth = ground_truth

    def Labels(self):
        return self.labels

    def Location(self):
        return self.location

    def GroundTruth(self):
        return self.ground_truth



# find the candidates for these prefixes, threshold and distance
def FindCandidates(prefix_one, prefix_two, threshold, maximum_distance):
    # get the gold decisions for this arrangement
    ground_truth = ReadGold(prefix_one, prefix_two, threshold, maximum_distance)
    ncandidates = len(ground_truth)

    # get the features for this arrangment
    labels, locations = ReadFeatures(prefix_one, prefix_two, threshold, maximum_distance)

    positive_candidates = []
    negative_candidates = []

    # only consider locations where there is legitimate ground truth
    for iv in range(ncandidates):
        # add the positive and negative candidates
        if ground_truth[iv] == 0:
            positive_candidates.append(Candidate(labels[iv], locations[iv], True))
        elif ground_truth[iv] == 1:
            negative_candidates.append(Candidate(labels[iv], locations[iv], False))

    candidates = []
    for iv in range(min(len(positive_candidates), len(negative_candidates))):
        candidates.append(positive_candidates[iv])
        candidates.append(negative_candidates[iv])

    return candidates



@jit(nopython=True)
def ScaleFeature(segmentation_one, segmentation_two, image_one, image_two, labels, position_one, position_two, radii, width, nchannels):
    # create the feature
    feature = np.zeros((1, width[IB_Z], width[IB_Y], width[IB_X], nchannels), dtype=np.uint8)

    # put the data into a more convenient form
    label_one, label_two = labels
    zres, yres, xres = segmentation_one.shape

    # iterate over the new window
    for iz in range(width[IB_Z]):
        for iy in range(width[IB_Y]):
            for ix in range(width[IB_X]):
                # get the offset
                zoffset = int(iz * float(2 * radii[IB_Z]) / width[IB_Z] + 0.5)
                yoffset = int(iy * float(2 * radii[IB_Y]) / width[IB_Y] + 0.5)
                xoffset = int(ix * float(2 * radii[IB_X]) / width[IB_X] + 0.5)

                # get the locations for both segments
                location_one = (position_one[IB_Z] + zoffset, position_one[IB_Y] + yoffset, position_one[IB_X] + xoffset)
                location_two = (position_two[IB_Z] + zoffset, position_two[IB_Y] + yoffset, position_two[IB_X] + xoffset)

                if location_one[IB_Z] > 0 and location_one[IB_Z] < zres and location_one[IB_Y] > 0 and location_one[IB_Y] < yres and location_one[IB_X] > 0 and location_one[IB_X] < xres:
                    if segmentation_one[location_one[IB_Z], location_one[IB_Y], location_one[IB_X]] == label_one:
                        feature[0,iz,iy,ix,0] = 1
                        feature[0,iz,iy,ix,2] = 1
                    #feature[0,iz,iy,ix,3] = image_one[location_one[IB_Z], location_one[IB_Y], location_one[IB_X]]
                if location_two[IB_Z] > 0 and location_two[IB_Z] < zres and location_two[IB_Y] > 0 and location_two[IB_Y] < yres and location_two[IB_X] > 0 and location_two[IB_X] < xres:
                    if segmentation_two[location_two[IB_Z], location_two[IB_Y], location_two[IB_X]] == label_two:
                        feature[0,iz,iy,ix,1] = 1
                        feature[0,iz,iy,ix,2] = 1
                    #feature[0,iz,iy,ix,3] = image_two[location_two[IB_Z], location_two[IB_Y], location_two[IB_X]]

    return feature



def ExtractFeature(segmentation_one, segmentation_two, image_one, image_two, bbox_one, bbox_two, candidate, radii, width, rotation, nchannels):
    # just checking
    assert (rotation < 16)

    # put the data into a more convenient form
    candidate_labels = candidate.Labels()
    candidate_location = candidate.Location()

    candidate_location = (candidate_location[IB_Z] - radii[IB_Z], candidate_location[IB_Y] - radii[IB_Y], candidate_location[IB_X] - radii[IB_X])

    # get the grid location from the corner
    position_one = WorldToGrid(candidate_location, bbox_one)
    position_two = WorldToGrid(candidate_location, bbox_two)

    # get this example
    example = ScaleFeature(segmentation_one, segmentation_two, image_one, image_two, candidate_labels, position_one, position_two, radii, width, nchannels)

    # should we flip the x axis
    flip_xaxis = rotation % 2
    flip_zaxis = (rotation / 2) % 2
    rotate_towards_Y = rotation / 4

    example = np.rot90(example, k=rotate_towards_Y, axes=(3,2))
    if flip_xaxis:
        example = np.flip(example, 3)
    if flip_zaxis:
        example = np.flip(example, 1)

    return example



@jit(nopython=True)
def CollapseFeature(example):
    # collapse three channels into one
    output = np.zeros((example.shape[IB_Z], example.shape[IB_Y], example.shape[IB_X]), dtype=np.uint8)

    zres, yres, xres, _ = example.shape

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if example[iz,iy,ix,0] and example[iz,iy,ix,1]:
                    output[iz,iy,ix] = 3
                elif example[iz,iy,ix,1]:
                    output[iz,iy,ix] = 2
                elif example[iz,iy,ix,0]:
                    output[iz,iy,ix] = 1

    return output




def SaveFeatures(prefix_one, prefix_two, threshold, maximum_distance, nchannels):
    # read in both segmentation and image files
    segmentation_one = dataIO.ReadSegmentationData(prefix_one)
    segmentation_two = dataIO.ReadSegmentationData(prefix_two)
    assert (segmentation_one.shape == segmentation_two.shape)
    image_one = dataIO.ReadImageData(prefix_one)
    image_two = dataIO.ReadImageData(prefix_two)
    bbox_one = dataIO.GetWorldBBox(prefix_one)
    bbox_two = dataIO.GetWorldBBox(prefix_two)
    world_res = dataIO.Resolution(prefix_one)
    assert (world_res == dataIO.Resolution(prefix_two))

    # get all of the candidates for these prefixes
    candidates = FindCandidates(prefix_one, prefix_two, threshold, maximum_distance)
    ncandidates = len(candidates)

    # get the radii for this feature
    radii = (maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X])
    width = (2 * radii[IB_Z], 2 * radii[IB_Y], 2 * radii[IB_X])

    for iv, candidate in enumerate(candidates):
        # with rotation equal to zero
        example = (ExtractFeature(segmentation_one, segmentation_two, image_one, image_two, bbox_one, bbox_two, candidate, radii, width, 0, nchannels))[0,:,:,:,:]

        # collapse the three channels into one
        example = CollapseFeature(example)

        # get the output filename
        filename = 'features/ebro/{}-{}/{}-{}nm-{:05d}.h5'.format(prefix_one, prefix_two, threshold, maximum_distance, iv)

        # write the h5 file
        dataIO.WriteH5File(example, filename, 'main')