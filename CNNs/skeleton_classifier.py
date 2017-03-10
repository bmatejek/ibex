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
        potential_merges, = struct.unpack('Q', fd.read(8))
        
        # create an array for merge candidates
        merge_candidates = []

        # iterate over all potential merge candidates
        for im in range(potential_merges):
            index_one, index_two, label_one, label_two, xpoint, ypoint, zpoint, ground_truth, = struct.unpack('QQQQQQQB', fd.read(57))

            merge_candidates.append(MergeCandidate(label_one, label_two, xpoint, ypoint, zpoint, ground_truth))

    return merge_candidates

@jit(nopython=True)
def make_window(segmentation, label_one, label_two, x, y, z, label, width):
    # TODO fix hardcoded values
    xradius, yradius, zradius = (100, 100, 13)

    segment = segmentation[z-zradius:z+zradius,y-yradius:y+yradius,x-xradius:x+xradius]
    (zres, yres, xres) = segment.shape

    example = np.zeros((width, width, width), dtype=np.uint8)

    for iz in range(width):
        for iy in range(width):
            for ix in range(width):
                segmentz = int(float(zres) / float(width) * iz)
                segmenty = int(float(yres) / float(width) * iy)
                segmentx = int(float(xres) / float(width) * ix)

                if segment[segmentz, segmenty, segmentx] == label_one or segment[segmentz, segmenty, segmentx] == label_two:
                    example[iz,iy,ix] = 1
                else:
                    example[iz,iy,ix] = 0

    return example, label


def data_generator(args, prefix):
    # read in h5 file
    filename = 'rhoana/' + prefix + '_rhoana.h5'
    segmentation = dataIO.ReadH5File(filename, 'main')

    # read in potential merge locations
    merge_filename = 'skeletons/' + prefix + '_merge_candidates.merge'
    merge_candidates = ReadMergeFilename(merge_filename)

    num_locations = len(merge_candidates)
    batch_num = 0
    while 1:
        # create empty examples and labels arrays
        examples = np.zeros((args.batch_size, args.window_width, args.window_width, args.window_width, 1))
        labels = np.zeros((args.batch_size, 2))

        # populate the examples and labels array
        for index in range(args.batch_size):
            total = index + batch_num * args.batch_size

            # get this merge candidate
            merge_candidate = merge_candidates[total]

            # make the window given the merge candidate
            window, label = make_window(segmentation, merge_candidate.label_one, merge_candidate.label_two, merge_candidate.x, merge_candidate.y, merge_candidate.z, merge_candidate.ground_truth, args.window_width)

            # update the input vectors
            examples[index,:,:,:,0] = window
            labels[index,:] = label

        # restart the batch number if needed
        batch_num += 1
        if (batch_num + 1) * args.batch_size > num_locations:
            batch_num = 0

        # return the current examples and labels
        yield (examples, labels)