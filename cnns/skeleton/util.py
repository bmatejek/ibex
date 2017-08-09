import struct
import numpy as np
from numba import jit
from ibex.utilities import dataIO
from ibex.utilities.constants import *


# class that contains all import feature data
class SkeletonCandidate:
    def __init__(self, labels, location, ground_truth):
        self.labels = labels
        self.location = location
        self.ground_truth = ground_truth



# find the candidates for this prefix and distance
def FindCandidates(prefix, threshold, maximum_distance, inference=False):
    if inference:
        filename = 'features/skeleton/{}-{}-{}nm-inference.candidates'.format(prefix, threshold, maximum_distance)
    else:
        filename = 'features/skeleton/{}-{}-{}nm-learning.candidates'.format(prefix, threshold, maximum_distance)

    # read the candidate filename
    with open(filename, 'rb') as fd:
        ncandidates, = struct.unpack('i', fd.read(4))
        candidates = []
        # iterate over all of the candidate merge locations
        for _ in range(ncandidates):
            label_one, label_two, zpoint, ypoint, xpoint, ground_truth = struct.unpack('QQQQQQ', fd.read(48))
            candidates.append(SkeletonCandidate((label_one, label_two), (zpoint, ypoint, xpoint), ground_truth))

    return candidates



@jit(nopython=True)
def ScaleSegment(segment, width, labels):
    # get the size of the larger segment
    zres, yres, xres = segment.shape
    label_one, label_two = labels
    nchannels = width[3]

    # create the example to be returned
    example = np.zeros((1, width[IB_Z], width[IB_Y], width[IB_X], nchannels), dtype=np.uint8)

    # iterate over the example coordinates
    for iz in range(width[IB_Z]):
        for iy in range(width[IB_Y]):
            for ix in range(width[IB_X]):
                # get the global coordiantes from segment
                iw = int(float(zres) / float(width[IB_Z]) * iz)
                iv = int(float(yres) / float(width[IB_Y]) * iy)
                iu = int(float(xres) / float(width[IB_X]) * ix)

                if nchannels == 1:
                    if segment[iw,iv,iu] == label_one or segment[iw,iv,iu] == label_two:
                        example[0,iz,iy,ix,0] = 1
                else:
                    if segment[iw,iv,iu] == label_one:
                        example[0,iz,iy,ix,0] = 1
                        example[0,iz,iy,ix,2] = 1

                    elif segment[iw,iv,iu] == label_two:
                        example[0,iz,iy,ix,1] = 1
                        example[0,iz,iy,ix,2] = 1
                        
    return example



# extract the feature given the location and segmentation'
def ExtractFeature(segmentation, candidate, width, radii, rotation):
    assert (rotation < 16)

    # get the data in a more convenient form
    zradius, yradius, xradius = radii
    zpoint, ypoint, xpoint = candidate.location
    labels = candidate.labels


    # extract the small window from this segment
    example = segmentation[zpoint-zradius:zpoint+zradius,ypoint-yradius:ypoint+yradius,xpoint-xradius:xpoint+xradius]

    # rescale the segment
    example = ScaleSegment(example, width, labels)

    # flip x axis? -> add 1 because of extra filler channel
    if rotation % 2: example = np.flip(example, IB_X + 1)
    # flip z axis?
    if (rotation / 2) % 2: example = np.flip(example, IB_Z + 1)

    # rotate in y?
    yrotation = rotation / 4
    example = np.rot90(example, k=yrotation, axes=(IB_X + 1, IB_Y + 1))

    return example













# @jit(nopython=True)
# def CollapseSegment(segmentation, label_one, label_two):
#     # get the shape for this segment
#     zres, yres, xres = segmentation.shape

#     # create the output collapsed segment
#     segment = np.zeros((zres, yres, xres), dtype=np.uint8)

#     for iz in range(zres):
#         for iy in range(yres):
#             for ix in range(xres):
#                 if (segmentation[iz,iy,ix] == label_one):
#                     segment[iz,iy,ix] = 1
#                 elif (segmentation[iz,iy,ix] == label_two):
#                     segment[iz,iy,ix] = 2

#     return segment



# # save all the features as h5 files
# def SaveFeatures(prefix, maximum_distance):
#     # read in the segmentation file
#     segmentation = dataIO.ReadSegmentationData(prefix)

#     # get the grid size and the world resolution in (z, y, x)
#     grid_size = segmentation.shape
#     world_res = dataIO.ReadMetaData(prefix)

#     # get the radii for the bounding box in grid coordinates
#     candidates = FindCandidates(prefix, maximum_distance, forward=True)
#     ncandidates = len(candidates)

#     # get the radii for the bounding box
#     zradius, yradius, xradius = (maximum_distance / world_res[0], maximum_distance / world_res[1], maximum_distance / world_res[2])

#     # iterate over all candidates
#     for ic, candidate in enumerate(candidates):
#         # get the center point for this candidate
#         zpoint, ypoint, xpoint = (candidate.Z(), candidate.Y(), candidate.X())

#         # collapse the segment
#         segment = CollapseSegment(segmentation[zpoint-zradius:zpoint+zradius,ypoint-yradius:ypoint+yradius,xpoint-xradius:xpoint+xradius], candidate.LabelOne(), candidate.LabelTwo())

#         # get the output filename
#         filename = 'features/skeleton/{}/{:05d}-feature.h5'.format(prefix, ic)

#         # output the h5 file
#         dataIO.WriteH5File(segment, filename, 'main')

#     # save the ground truth
#     filename = 'features/skeleton/{}-ground-truth.txt'.format(prefix)

#     with open(filename, 'w') as fd:
#         for candidate in candidates:
#             fd.write(str(candidate.GroundTruth()) + '\n')