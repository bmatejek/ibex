import numpy as np

from numba import jit

from ibex.utilities.constants import *



@jit(nopython=True)
def FindSmallSegments(segmentation, threshold):
    # create lists for small and large nodes
    small_segments = set()
    large_segments = set()

    zres, yres, xres = segmentation.shape

    # create a count for each label
    max_label = np.amax(segmentation) + 1
    counts = np.zeros(max_label, dtype=np.int64)

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                counts[segmentation[iz,iy,ix]] += 1

    for label in range(max_label):
        if not counts[label]: continue

        if (counts[label] < threshold): small_segments.add(label)
        else: large_segments.add(label)

    return small_segments, large_segments




@jit(nopython=True)
def ScaleFeature(segment, width, label_one, label_two):
    # get the size of the extracted segment
    zres, yres, xres = segment.shape

    example = np.zeros((width[IB_Z], width[IB_Y], width[IB_X]), dtype=np.int8)

    # iterate over the example coordinates
    for iz in range(width[IB_Z]):
        for iy in range(width[IB_Y]):
            for ix in range(width[IB_X]):
                # get the global coordiantes from segment
                iw = int(float(zres) / float(width[IB_Z]) * iz)
                iv = int(float(yres) / float(width[IB_Y]) * iy)
                iu = int(float(xres) / float(width[IB_X]) * ix)

                if segment[iw,iv,iu] == label_one:
                    example[iz,iy,ix] = 1
                elif segment[iw,iv,iu] == label_two:
                    example[iz,iy,ix] = 2

    return example




@jit(nopython=True)
def ExtractExample(segment, label_one, label_two):
    zres, yres, xres = segment.shape

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if (not segment[iz,iy,ix] == label_one) and (not segment[iz,iy,ix] == label_two):
                    segment[iz,iy,ix] = 0

    return segment
