import struct
from numba import jit
import numpy as np


# class that contains all import feature data
class Candidate:
    def __init__(self, label_one, label_two, location, ground_truth):
        self.label_one = label_one
        self.label_two = label_two
        self.location = location
        self.ground_truth = ground_truth

    def LabelOne(self):
        return self.label_one

    def LabelTwo(self):
        return self.label_two

    def Labels(self):
        return (self.label_one, self.label_two)

    def Location(self):
        return self.location

    def X(self):
        return self.location[2]

    def Y(self):
        return self.location[1]

    def Z(self):
        return self.location[0]

    def GroundTruth(self):
        return self.ground_truth



# find the candidates for this prefix and distance
def FindCandidates(prefix, maximum_distance, forward=False):
    if forward: filename = 'skeletons/{0}_{1}nm_forward.candidates'.format(prefix, maximum_distance)
    else: filename = 'skeletons/{0}_{1}nm_train.candidates'.format(prefix, maximum_distance)

    # read the candidate filename
    with open(filename, 'rb') as fd:
        ncandidates, = struct.unpack('I', fd.read(4))

        candidates = []

        # iterate over all of the candidate merge locations
        for _ in range(ncandidates):
            label_one, label_two, zpoint, ypoint, xpoint, ground_truth = struct.unpack('QQQQQQ', fd.read(48))

            candidates.append(Candidate(label_one, label_two, (zpoint, ypoint, xpoint), ground_truth))

    return candidates



@jit(nopython=True)
def ScaleSegment(segment, window_width, labels):
    # get the size of the larger segment
    zres, yres, xres = segment.shape
    label_one, label_two = labels

    # create the example to be returned
    example = np.zeros((1, window_width, window_width, window_width, 1), dtype=np.uint8)

    # iterate over the example coordinates
    for iz in range(window_width):
        for iy in range(window_width):
            for ix in range(window_width):
                # get the global coordiantes from segment
                iw = int(float(zres) / float(window_width) * iz)
                iv = int(float(yres) / float(window_width) * iy)
                iu = int(float(xres) / float(window_width) * ix)

                if segment[iw,iv,iu] == label_one:
                    example[0,iz,iy,ix,0] = 1
                elif segment[iw,iv,iu] == label_two:
                    example[0,iz,iy,ix,0] = 1
                else:
                    example[0,iz,iy,ix,0] = 0

    return example



# extract the feature given the location and segmentation'
def ExtractFeature(segmentation, labels, location, radii, window_width, rotation=0):
    # get the data in a more convenient form
    zradius, yradius, xradius = radii
    zpoint, ypoint, xpoint = location

    # extract the small window from this segment
    segment = segmentation[zpoint-zradius:zpoint+zradius,ypoint-yradius:ypoint+yradius,xpoint-xradius:xpoint+xradius]

    # rotate the segment
    if rotation == 1: segment = np.flip(segment, 0)
    elif rotation == 2: segment = np.flip(segment, 1)
    elif rotation == 3: segment = np.flip(segment, 2)
    elif rotation == 4: segment = np.flip(np.flip(segment, 0), 1)
    elif rotation == 5: segment = np.flip(np.flip(segment, 0), 2)
    elif rotation == 6: segment = np.flip(np.flip(segment, 1), 2)
    elif rotation == 7: segment = np.flip(np.flip(np.flip(segment, 0,), 1), 2)

    return ScaleSegment(segment, window_width, labels)
