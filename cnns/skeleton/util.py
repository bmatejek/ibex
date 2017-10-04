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
def FindCandidates(prefix, threshold, maximum_distance, network_distance, inference=False):
    if inference:
        filename = 'features/skeleton/{}-{}-{}nm-{}nm-inference.candidates'.format(prefix, threshold, maximum_distance, network_distance)
    else:
        filename = 'features/skeleton/{}-{}-{}nm-{}nm-learning.candidates'.format(prefix, threshold, maximum_distance, network_distance)

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

                if nchannels == 1 and (segment[iw,iv,iu] == label_one or segment[iw,iv,iu] == label_two):
                        example[0,iz,iy,ix,0] = 1
                else:
                    # add second channel
                    if segment[iw,iv,iu] == label_one:
                        example[0,iz,iy,ix,0] = 1
                    elif segment[iw,iv,iu] == label_two:
                        example[0,iz,iy,ix,1] = 1
                    # add third channel 
                    if nchannels == 3 and (segment[iw,iv,iu] == label_one or segment[iw,iv,iu] == label_two):
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



# save the features for viewing
def SaveFeatures(prefix, threshold, maximum_distance):
    # read in relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    grid_size = segmentation.shape
    world_res = dataIO.Resolution(prefix)

    # get the radii for the bounding box
    radii = (maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X])
    width = (2 * radii[IB_Z], 2 * radii[IB_Y], 2 * radii[IB_X], 3)

    # read all candidates
    candidates = FindCandidates(prefix, threshold, maximum_distance, inference=True)    
    ncandidates = len(candidates)

    for iv, candidate in enumerate(candidates):
        # get an example with zero rotation
        example = ExtractFeature(segmentation, candidate, width, radii, 0)

        # compress the channels
        compressed_output = np.zeros((width[IB_Z], width[IB_Y], width[IB_X]), dtype=np.uint8)
        compressed_output[example[0,:,:,:,0] == 1] = 1
        compressed_output[example[0,:,:,:,1] == 1] = 2

        # save the output file
        filename = 'features/skeleton/{}/{}-{}nm-{:05d}.h5'.format(prefix, threshold, maximum_distance, iv)
        dataIO.WriteH5File(compressed_output, filename, 'main')
