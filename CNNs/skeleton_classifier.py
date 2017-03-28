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
    def __init__(self, label_one, label_two, x, y, z, ground_truth, rotation):
        self.label_one = label_one
        self.label_two = label_two
        self.x = x
        self.y = y
        self.z = z
        self.ground_truth = ground_truth
        self.rotation = rotation

def ReadMergeFilename(filename):
    with open(filename, 'rb') as fd:
        # read the number of potential merges
        npositives, = struct.unpack('Q', fd.read(8))
        nnegatives, = struct.unpack('Q', fd.read(8))
        xradius, yradius, zradius, = struct.unpack('QQQ', fd.read(24))

        # create an array for merge candidates
        merge_candidates = []

        # iterate over all potential merge candidates
        for im in range(npositives + nnegatives):
            label_one, label_two, xpoint, ypoint, zpoint, ground_truth, rotation, = struct.unpack('QQQQQBB', fd.read(42))

            merge_candidates.append(MergeCandidate(label_one, label_two, xpoint, ypoint, zpoint, ground_truth, rotation))

    return merge_candidates, npositives, nnegatives, (xradius, yradius, zradius)

@jit(nopython=True)
def make_window(segmentation, label_one, label_two, x, y, z, radii, width):
    xradius, yradius, zradius = radii

    segment = segmentation[z-zradius:z+zradius,y-yradius:y+yradius,x-xradius:x+xradius]
    (zres, yres, xres) = segment.shape

    example = np.zeros((width, width, width, 1), dtype=np.uint8)

    for iz in range(width):
        for iy in range(width):
            for ix in range(width):
                iw = int(float(zres) / float(width) * iz)
                iv = int(float(yres) / float(width) * iy)
                iu = int(float(xres) / float(width) * ix)

                if segment[iw,iv,iu] == label_one:
                    example[iz,iy,ix,:] = 1
                elif segment[iw,iv,iu] == label_two:
                    example[iz,iy,ix,:] = 1
                else:
                    example[iz,iy,ix,:] = 0


    return example

def apply_rotation(example, rotation):
    # apply some rotation
    if rotation == 0: return example
    elif rotation == 1: return np.flip(example, 0)
    elif rotation == 2: return np.flip(example, 1)
    elif rotation == 3: return np.flip(example, 2)
    elif rotation == 4: return np.flip(np.flip(example, 0), 1)
    elif rotation == 5: return np.flip(np.flip(example, 0), 2)
    elif rotation == 6: return np.flip(np.flip(example, 1), 2)
    elif rotation == 7: return np.flip(np.flip(np.flip(example, 0), 1), 2)
