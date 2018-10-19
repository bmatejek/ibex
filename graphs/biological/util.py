import numpy as np

from numba import jit

from ibex.utilities.constants import *

@jit(nopython=True)
def FindMiddleBoundaries(segmentation):
    zres, yres, xres = segmentation.shape

    max_label = np.amax(segmentation) + 1

    zmean = np.zeros((max_label, max_label), dtype=np.float32)
    ymean = np.zeros((max_label, max_label), dtype=np.float32)
    xmean = np.zeros((max_label, max_label), dtype=np.float32)
    counts = np.zeros((max_label, max_label), dtype=np.float32)

    zdiff = segmentation[1:,:,:] != segmentation[:-1,:,:]
    ydiff = segmentation[:,1:,:] != segmentation[:,:-1,:]
    xdiff = segmentation[:,:,1:] != segmentation[:,:,:-1]

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if iz < zres - 1 and zdiff[iz,iy,ix]:
                    label_one = min(segmentation[iz,iy,ix], segmentation[iz+1,iy,ix])
                    label_two = max(segmentation[iz,iy,ix], segmentation[iz+1,iy,ix])  
                    zmean[label_one,label_two] += (iz + 0.5)
                    ymean[label_one,label_two] += iy
                    xmean[label_one,label_two] += ix
                    counts[label_one,label_two] += 1
                    
                if iy < yres - 1 and ydiff[iz,iy,ix]:
                    label_one = min(segmentation[iz,iy,ix], segmentation[iz,iy+1,ix])
                    label_two = max(segmentation[iz,iy,ix], segmentation[iz,iy+1,ix])
                    zmean[label_one,label_two] += iz
                    ymean[label_one,label_two] += (iy + 0.5)
                    xmean[label_one,label_two] += ix
                    counts[label_one,label_two] += 1
                    
                if ix < xres - 1 and xdiff[iz,iy,ix]:
                    label_one = min(segmentation[iz,iy,ix], segmentation[iz,iy,ix+1])
                    label_two = max(segmentation[iz,iy,ix], segmentation[iz,iy,ix+1])
                    zmean[label_one,label_two] += iz
                    ymean[label_one,label_two] += iy
                    xmean[label_one,label_two] += (ix + 0.5)
                    counts[label_one,label_two] += 1

    for is1 in range(max_label):
        for is2 in range(is1 + 1, max_label):
            if not counts[is1,is2]: continue 
            zmean[is1,is2] /= counts[is1,is2]
            ymean[is1,is2] /= counts[is1,is2]
            xmean[is1,is2] /= counts[is1,is2]

            zmean[is2,is1] = zmean[is1,is2]
            ymean[is2,is1] = ymean[is1,is2]
            xmean[is2,is1] = xmean[is1,is2]
            counts[is2,is1] = counts[is1,is2]

    return zmean, ymean, xmean


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
