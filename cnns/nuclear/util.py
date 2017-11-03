import struct
import numpy as np

from numba import jit
from ibex.utilities import dataIO
from ibex.utilities.constants import *



# class that contains all import feature data
class NuclearCandidate:
    def __init__(self, label, location, ground_truth):
        self.label = label
        self.location = location
        self.ground_truth = ground_truth



# find the candidates for this prefix and distance
def FindCandidates(prefix, threshold, network_radius, inference=False, validation=False):
    if inference:
        filename = 'features/nuclear/{}-{}-{}nm-inference.candidates'.format(prefix, threshold, network_radius)
    elif validation:
        filename = 'features/nuclear/{}-{}-{}nm-validation.candidates'.format(prefix, threshold, network_radius)
    else:
        filename = 'features/nuclear/{}-{}-{}nm-training.candidates'.format(prefix, threshold, network_radius)

    # read the candidate filename
    with open(filename, 'rb') as fd:
        ncandidates, = struct.unpack('i', fd.read(4))
        candidates = []
        # iterate over all of the candidate merge locations
        for _ in range(ncandidates):
            label, zpoint, ypoint, xpoint, ground_truth = struct.unpack('QQQQQ', fd.read(40))
            candidates.append(NuclearCandidate(label, (zpoint, ypoint, xpoint), ground_truth))

    return candidates



@jit(nopython=True)
def ScaleSegment(segment, width, label):
    # get the size of the largest segment
    zres, yres, xres = segment.shape
    nchannels = width[0]

    # create the example to be returned
    example = np.zeros((1, nchannels, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1]), dtype=np.float32)

    # iterate over the example coordinates
    for iz in range(width[IB_Z] + 1):
        for iy in range(width[IB_Y] + 1):
            for ix in range(width[IB_X] + 1):
                # get the global coordinates from segment
                iw = int(float(zres) / float(width[IB_Z + 1]) * iz)
                iv = int(float(yres) / float(width[IB_Y + 1]) * iy)
                iu = int(float(xres) / float(width[IB_X + 1]) * ix)

                if segment[iw,iv,iu] == label: example[0,0,iz,iy,ix] = 1

    example = example - 0.5

    return example



# extract the feature given the location and segmentation
def ExtractFeature(segmentation, candidate, width, radii, rotation):
    assert (rotation < 16)

    # get the data in a more convenient way
    zradius, yradius, xradius = radii
    zpoint, ypoint, xpoint = candidate.location
    label = candidate.label

    # extract the small window from this segment
    example = segmentation[zpoint-zradius:zpoint+zradius+1,ypoint-yradius:ypoint+yradius+1,xpoint-xradius:xpoint+xradius+1]

    # rescale the segment
    example = ScaleSegment(example, width, label)

    # flip x axis? -> add 2 because of extra filler channel
    if rotation % 2: example = np.flip(example, IB_X + 2)
    # flip z axis?
    if (rotation / 2) % 2: example = np.flip(example, IB_Z + 2)
    
    # rotate in y?
    yrotation = rotation / 4
    example = np.rot90(example, k=yrotation, axes=(IB_X + 2, IB_Y + 2))

    return example