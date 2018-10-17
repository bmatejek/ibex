import os
import numpy as np
from numba import jit

from ibex.graphs.biological import edge_generation
from ibex.utilities import dataIO
from ibex.utilities.constants import *
from ibex.data_structures import unionfind;
from ibex.transforms import seg2seg



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
def FindMiddleBoundary(segmentation):
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
def ExtractExample(segment, label_one, label_two):
    zres, yres, xres = segment.shape

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if segment[iz,iy,ix] == label_one:
                    segment[iz,iy,ix] = 1
                elif segment[iz,iy,ix] == label_two:
                    segment[iz,iy,ix] = 2
                else:
                    segment[iz,iy,ix] = 0

    return segment
    


def GenerateNodes(prefix, segmentation, seg2gold_mapping, threshold=20000, radius=600, subset='training'):
    # make sure directory structure exists
    sub_directory = 'features/biological/nodes-{}nm'.format(radius)
    if not os.path.exists(sub_directory):
        os.mkdir(sub_directory)
    sub_directory = '{}/{}'.format(sub_directory, subset)
    if not os.path.exists(sub_directory):
        os.mkdir(sub_directory)
    labelings = ['positives', 'negatives', 'unknowns']
    for labeling in labelings:
        if not os.path.exists('{}/{}'.format(sub_directory, labeling)):
            os.mkdir('{}/{}'.format(sub_directory, labeling))

    # get the complete adjacency graph shows all neighboring edges
    adjacency_graph = edge_generation.ExtractAdjacencyMatrix(segmentation)

    # get the list of nodes over and under 20K
    small_segments, large_segments = FindSmallSegments(segmentation, threshold)

    # get the locations around a possible merge
    zmean, ymean, xmean = FindMiddleBoundary(segmentation)

    # get the radius along each dimensions in terms of voxels
    resolution = dataIO.Resolution(prefix)
    (zradius, yradius, xradius) = (int(radius / resolution[IB_Z]), int(radius / resolution[IB_Y]), int(radius / resolution[IB_X]))
    zres, yres, xres = segmentation.shape

    if subset == 'training' or subset == 'validation':
        ((cropped_zmin, cropped_zmax), (cropped_ymin, cropped_ymax), (cropped_xmin, cropped_xmax)) = dataIO.CroppingBox(prefix)
    elif subset == 'testing':
        ((cropped_zmin, cropped_zmax), (cropped_ymin, cropped_ymax), (cropped_xmin, cropped_xmax)) = ((0, zres), (0, yres), (0, xres))
    else:
        sys.stderr.write('Unrecognized subset: {}'.format(subset))

    old_segmentation = np.copy(segmentation)

    for iv, (label_one, label_two) in enumerate(adjacency_graph):
        if (label_one in small_segments) ^ (label_two in small_segments):
            zpoint = int(zmean[label_one,label_two])
            ypoint = int(ymean[label_one,label_two])
            xpoint = int(xmean[label_one,label_two])

            # if the center of the point falls outside the cropped box do not include it 
            if (zpoint < cropped_zmin or cropped_zmax <= zpoint): continue
            if (ypoint < cropped_ymin or cropped_ymax <= ypoint): continue
            if (xpoint < cropped_xmin or cropped_xmax <= xpoint): continue

            # need to make sure that bounding box does not leave location so sizes are correct
            zmin = max(0, zpoint - zradius)
            ymin = max(0, ypoint - yradius)
            xmin = max(0, xpoint - xradius)
            zmax = min(zres, zpoint + zradius + 1)
            ymax = min(yres, ypoint + yradius + 1)
            xmax = min(xres, xpoint + xradius + 1)

            # create the empty example file with three channels corresponding to the value of segment
            example = np.zeros((2 * zradius + 1, 2 * yradius + 1, 2 * xradius + 1), dtype=np.int32)

            # get the valid location around this point
            segment = ExtractExample(segmentation[zmin:zmax,ymin:ymax,xmin:xmax].copy(), label_one, label_two)

            if example.shape == segment.shape:
                example = segment
            else:
                if zmin == 0: zstart = zradius - zpoint
                else: zstart = 0

                if ymin == 0: ystart = yradius - ypoint
                else: ystart = 0

                if xmin == 0: xstart = xradius - xpoint
                else: xstart = 0

                # the second and third channels are one if the corresponding voxels belong to the individual segments
                example[zstart:zstart+segment.shape[IB_Z],ystart:ystart+segment.shape[IB_Y],xstart:xstart+segment.shape[IB_X]] = segment

            # see if these two segments belong to the same node
            gold_one = seg2gold_mapping[label_one]
            gold_two = seg2gold_mapping[label_two]

            # save the data in the appropriate location
            if gold_one < 1 or gold_two < 1: 
                output_directory = 'features/biological/nodes-{}nm/{}/unknowns'.format(radius, subset)
            elif gold_one == gold_two:
                output_directory = 'features/biological/nodes-{}nm/{}/positives'.format(radius, subset)
            else: 
                output_directory = 'features/biological/nodes-{}nm/{}/negatives'.format(radius, subset)
            output_filename = '{}/{}-{}-{}.h5'.format(output_directory, prefix, label_one, label_two)

            # write this example
            dataIO.WriteH5File(example, output_filename, 'main')