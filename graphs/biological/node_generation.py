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
def ScaleSegment(segment, width, labels):
    zres, yres, xres = segment.shape
    label_one, label_two = labels
    
    example = np.zeros((width[IB_Z], width[IB_Y], width[IB_X]), dtype=np.float32)
    
    for iz in range(width[IB_Z]):
        for iy in range(width[IB_Y]):
            for ix in range(width[IB_X]):
                # get the global coordinates from segment
                iw = int(float(zres) / float(width[IB_Z]) * iz)
                iv = int(float(yres) / float(width[IB_Y]) * iy)
                iu = int(float(xres) / float(width[IB_X]) * ix)
                
                if segment[iw,iv,iu] == label_one or segment[iw,iv,iu] == label_two:
                    example[iz,iy,ix] = 1
                    
    return example



def GenerateNodes(prefix, segmentation, seg2gold_mapping, threshold=20000, radius=600, subset='training'):
    # make sure directory structure exists
    sub_directory = 'features/biological/nodes'
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

    resolution = [40, 3.6, 3.6] #dataIO.Resolution(prefix)
    radii = (int(radius / resolution[IB_Z]), int(radius / resolution[IB_Y]), int(radius / resolution[IB_X]))

    # hardcoded for now
    width = (18, 52, 52)

    zradius, yradius, xradius = radii
    zres, yres, xres = segmentation.shape

    for (label_one, label_two) in adjacency_graph:
        if (label_one in small_segments) ^ (label_two in small_segments):
            zpoint = int(zmean[label_one,label_two])
            ypoint = int(ymean[label_one,label_two])
            xpoint = int(xmean[label_one,label_two])

            zmin = max(0, zpoint - zradius)
            ymin = max(0, ypoint - yradius)
            xmin = max(0, xpoint - xradius)
            zmax = min(zres, zpoint + zradius + 1)
            ymax = min(yres, ypoint + yradius + 1)
            xmax = min(xres, xpoint + xradius + 1)

            example = np.zeros((2 * zradius + 1, 2 * yradius + 1, 2 * xradius + 1), dtype=np.int32)
            segment = segmentation[zmin:zmax,ymin:ymax,xmin:xmax]

            if example.shape == segment.shape:
                example = segment
            else:
                if zmin == 0: zstart = zradius - zpoint
                else: zstart = 0

                if ymin == 0: ystart = yradius - ypoint
                else: ystart = 0

                if xmin == 0: xstart = xradius - xpoint
                else: xstart = 0

                example[zstart:zstart+segment.shape[IB_Z],ystart:ystart+segment.shape[IB_Y],xstart:xstart+segment.shape[IB_X]] = segment

            example = ScaleSegment(example, width, (label_one, label_two))

            gold_one = seg2gold_mapping[label_one]
            gold_two = seg2gold_mapping[label_two]

            if gold_one < 1 or gold_two < 1: 
                output_directory = 'features/biological/nodes/{}/unknowns'.format(subset)
            elif gold_one == gold_two:
                output_directory = 'features/biological/nodes/{}/positives'.format(subset)
            else: 
                output_directory = 'features/biological/nodes/{}/negatives'.format(subset)

            # save this example
            output_filename = '{}/{}-{}-{}.h5'.format(output_directory, prefix, label_one, label_two)

            dataIO.WriteH5File(example, output_filename, 'main')