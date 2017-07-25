import struct
from ibex.utilities.constants import *
from numba import jit
import numpy as np


# go from world coordinates to grid coordinates
def WorldToGrid(world_position, bounding_box):
    zdiff = world_position[IB_Z] - bounding_box.Min(IB_Z)
    ydiff = world_position[IB_Y] - bounding_box.Min(IB_Y)
    xdiff = world_position[IB_X] - bounding_box.Min(IB_X)

    return (zdiff, ydiff, xdiff)



# class that contains all import feature data
class Candidate:
    def __init__(self, label_one, label_two, location, ground_truth):
        self.label_one = label_one
        self.label_two = label_two
        self.location = location
        self.ground_truth = ground_truth

    def Labels(self):
        return (self.label_one, self.label_two)

    def Location(self):
        return self.location[IB_Z], self.location[IB_Y], self.location[IB_X]

    def GroundTruth(self):
        return self.ground_truth



# find the candidates for these prefixes, threshold and distance
def FindCandidates(prefix_one, prefix_two, threshold, maximum_distance):
    # get feature filename
    feature_filename = 'features/ebro/{}-{}-{}-{}nm.candidates'.format(prefix_one, prefix_two, threshold, maximum_distance)
    # get gold filename
    gold_filename = 'gold/{}-{}-{}-{}nm.gold'.format(prefix_one, prefix_two, threshold, maximum_distance)

    # open the two files
    feature_fd = open(feature_filename, 'rb')
    gold_fd = open(gold_filename, 'rb')

    _, = struct.unpack('Q', feature_fd.read(8))
    ncandidates, = struct.unpack('I', gold_fd.read(4))

    positive_candidates = []
    negative_candidates = []
    for iv in range(ncandidates):
        label_one, label_two, centerx, centery, centerz, = struct.unpack('QQQQQ', feature_fd.read(40))
        decision, = struct.unpack('I', gold_fd.read(4))

        # merge this candidate
        if decision == 0:
            positive_candidates.append(Candidate(label_one, label_two, (centerz, centery, centerx), True))
        # do not merge this candidate
        elif decision == 1:
            negative_candidates.append(Candidate(label_one, label_two, (centerz, centery, centerx), False))
        # undecided
        else:
            continue

    # close the files
    feature_fd.close()
    gold_fd.close()

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