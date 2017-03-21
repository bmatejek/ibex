import argparse
import numpy as np
import struct
import sys
import os
from numba import jit

# add parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utilities import dataIO

class MergeCandidate():
    def __init__(self, label_one, label_two, x, y, z, ground_truth):
        self.label_one = label_one
        self.label_two = label_two
        self.x = x
        self.y = y
        self.z = z
        self.ground_truth = ground_truth

def ReadMergeFilename(filename):
    with open(filename, 'rb') as fd:
        # read the number of potential merges
        candidates, = struct.unpack('Q', fd.read(8))
        npositives, = struct.unpack('Q', fd.read(8))
        nnegatives, = struct.unpack('Q', fd.read(8))

        # create an array for merge candidates
        merge_candidates = []

        # iterate over all potential merge candidates
        for im in range(candidates):
            index_one, index_two, label_one, label_two, xpoint, ypoint, zpoint, ground_truth, = struct.unpack('QQQQQQQB', fd.read(57))

            merge_candidates.append(MergeCandidate(label_one, label_two, xpoint, ypoint, zpoint, ground_truth))

    return merge_candidates, npositives, nnegatives

@jit(nopython=True)
def make_window(segmentation, label_one, label_two, x, y, z, width):
    # TODO fix hardcoded values
    xradius, yradius, zradius = (100, 100, 13)

    segment = segmentation[z-zradius:z+zradius,y-yradius:y+yradius,x-xradius:x+xradius]
    (zres, yres, xres) = segment.shape

    example = np.zeros((width, width, width, 3), dtype=np.uint8)

    for iz in range(width):
        for iy in range(width):
            for ix in range(width):
                iw = int(float(zres) / float(width) * iz)
                iv = int(float(yres) / float(width) * iy)
                iu = int(float(xres) / float(width) * ix)

                if segment[iw,iv,iu] == label_one:
                    example[iz,iy,ix,0] = 1
                    example[iz,iy,ix,1] = 1
                    example[iz,iy,ix,2] = 0
                elif segment[iw,iv,iu] == label_two:
                    example[iz,iy,ix,0] = 1
                    example[iz,iy,ix,1] = 0
                    example[iz,iy,ix,2] = 1
                else:
                    example[iz,iy,ix,:] = 0

    return example