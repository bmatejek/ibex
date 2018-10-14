import math
import time
import random
import struct

import numpy as np
from numba import jit

from ibex.utilities import dataIO
from ibex.utilities.constants import *
from ibex.transforms import seg2gold
#from ibex.cnns.skeleton.util import SkeletonCandidate


# save the candidate files for the CNN
def SaveCandidates(output_filename, candidates):
    random.shuffle(candidates)
    ncandidates = len(candidates)

    # write all candidates to the file
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('q', len(candidates)))

        # add every candidate to the binary file
        for candidate in candidates:
            # get the labels for this candidate
            label_one = candidate.labels[0]
            label_two = candidate.labels[1]

            # get the location of this candidate
            position = candidate.location
            ground_truth = candidate.ground_truth

            # write this candidate to the evaluation candidate list
            fd.write(struct.pack('qqqqq?', label_one, label_two, position[IB_Z], position[IB_Y], position[IB_X], ground_truth))          




@jit(nopython=True)
def ExtractAdjacencyMatrix(segmentation):
    zres, yres, xres = segmentation.shape

    # create a set of neighbors as a tuple with the lower label first
    # if (z, y, x) is 1, the neighbor +1 is a different label
    xdiff = segmentation[:,:,1:] != segmentation[:,:,:-1]
    ydiff = segmentation[:,1:,:] != segmentation[:,:-1,:]
    zdiff = segmentation[1:,:,:] != segmentation[:-1,:,:]

    adjacency_graph = set()
    
    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if iz < zres - 1 and zdiff[iz,iy,ix]:
                    adjacency_graph.add((segmentation[iz,iy,ix], segmentation[iz+1,iy,ix]))
                if iy < yres - 1 and ydiff[iz,iy,ix]:
                    adjacency_graph.add((segmentation[iz,iy,ix], segmentation[iz,iy+1,ix]))
                if ix < xres - 1 and xdiff[iz,iy,ix]:
                    adjacency_graph.add((segmentation[iz,iy,ix], segmentation[iz,iy,ix+1]))

    # make sure that label_one is less than label_two to avoid double edges
    corrected_adjacency_graph = set()
    for (label_one, label_two) in adjacency_graph:
        if not label_one or not label_two: continue
        if label_two < label_one: corrected_adjacency_graph.add((label_two, label_one))
        else: corrected_adjacency_graph.add((label_one, label_two))

    return corrected_adjacency_graph


def BaselineGraph(segmentation, seg2gold_mapping):
    # get the adjacency matrix
    adjacency_graph = ExtractAdjacencyMatrix(segmentation)

    positive_candidates = []
    negative_candidates = []
    undetermined_candidates = []

    for (label_one, label_two) in adjacency_graph:
        gold_one = seg2gold_mapping[label_one]
        gold_two = seg2gold_mapping[label_two]

        if not gold_one or not gold_two: undetermined_candidates.append((label_one, label_two))
        elif gold_one == -1 or gold_two == -1: negative_candidates.append((label_one, label_two))
        elif gold_one == gold_two: positive_candidates.append((label_one, label_two))
        else: negative_candidates.append((label_one, label_two))

    print 'Baseline Adjacency Graph'
    print '  Number positive edges {}'.format(len(positive_candidates))
    print '  Number negative edges {}'.format(len(negative_candidates))
    print '  Number undetermined edges {}'.format(len(undetermined_candidates))



def GenerateEdges(prefix, segmentation, seg2gold_mapping, maximum_distance):
    start_time = time.time()

    # read in the skeletons for this prefix
    skeletons = dataIO.ReadSkeletons(prefix)
    max_label = len(skeletons)

    # read in the segmentation and gold datasets to find a mapping
    resolution = dataIO.Resolution(prefix)

    # keep track of all the potential locations
    midpoints = []
    ground_truths = []
    labels = []

    adjacency_graph = ExtractAdjacencyMatrix(segmentation)

    # go through all pairs of endpoints
    for is1 in range(max_label):
        skeleton_one = skeletons[is1]
        for is2 in range(is1 + 1, max_label):
            skeleton_two = skeletons[is2]

            if not (is1, is2) in adjacency_graph: continue

            # compare all endpoints between the two skeletons
            for endpoint_one in skeleton_one.endpoints:
                for endpoint_two in skeleton_two.endpoints:

                    # get the vectors of both angles
                    vector_one = endpoint_one.vector
                    vector_two = endpoint_two.vector

                    if (vector_one.dot(vector_two) > 0): continue

                    zdiff = resolution[IB_Z] * (endpoint_two.iz - endpoint_one.iz)
                    ydiff = resolution[IB_Y] * (endpoint_two.iy - endpoint_one.iy)
                    xdiff = resolution[IB_X] * (endpoint_two.ix - endpoint_one.ix)

                    # don't take sqrt, just square maximum distance
                    distance = zdiff * zdiff + ydiff * ydiff + xdiff * xdiff

                    if (distance < maximum_distance * maximum_distance):
                        midpoint = ((endpoint_two.iz + endpoint_one.iz) / 2, (endpoint_two.iy + endpoint_one.iy) / 2, (endpoint_two.ix + endpoint_one.ix) / 2)
                        ground_truth = (seg2gold_mapping[is1] == seg2gold_mapping[is2])
                        if not seg2gold_mapping[is1] and not seg2gold_mapping[is2]:
                            ground_truth = -1

                        midpoints.append(midpoint)
                        ground_truths.append(ground_truth)
                        labels.append((is1, is2))

    # create list of candidates
    positive_candidates = []
    negative_candidates = []
    undetermined_candidates = []

    for iv in range(len(midpoints)):
        if ground_truths[iv] == -1: 
            undetermined_candidates.append(SkeletonCandidate(labels[iv], midpoints[iv], ground_truths[iv]))
        elif ground_truths[iv] == 0: 
            negative_candidates.append(SkeletonCandidate(labels[iv], midpoints[iv], ground_truths[iv]))
        elif ground_truths[iv] == 1: 
            positive_candidates.append(SkeletonCandidate(labels[iv], midpoints[iv], ground_truths[iv]))

    print 'Number positive edges {}'.format(len(positive_candidates))
    print 'Number negative edges {}'.format(len(negative_candidates))
    print 'Number undetermined edges {}'.format(len(undetermined_candidates))

    # save positive and negative candidates separately
    positive_filename = 'features/skeleton/{}-{}nm-positive.candidates'.format(prefix, maximum_distance)
    negative_filename = 'features/skeleton/{}-{}nm-negative.candidates'.format(prefix, maximum_distance)
    undetermined_filename = 'features/skeleton/{}-{}nm-undetermined.candidates'.format(prefix, maximum_distance)
        
    SaveCandidates(positive_filename, positive_candidates)
    SaveCandidates(negative_filename, negative_candidates)
    SaveCandidates(undetermined_filename, undetermined_candidates)
    print time.time() - start_time

    # print time.time() - start_time


    # skeletons = dataIO.ReadSkeletons(prefix)

    # endpoints = [ skeleton.Endpoints2Array() for skeleton in skeletons.skeletons ]


    # # keep track of all the potential locations
    # midpoints = []
    # ground_truths = []
    # labels = []

    # # go through all pairs of skeletons and find endpoints within maximum_distance
    # max_label = len(endpoints)
    # for is1 in range(max_label):
    #     for is2 in range(is1 + 1, max_label):
    #         # go through all endpoints in segment one
    #         for endpoint_one in endpoints[is1]:
    #             for endpoint_two in endpoints[is2]:
    #                 zdiff = resolution[IB_Z] * (endpoint_two[IB_Z] - endpoint_one[IB_Z])
    #                 ydiff = resolution[IB_Y] * (endpoint_two[IB_Y] - endpoint_one[IB_Y])
    #                 xdiff = resolution[IB_X] * (endpoint_two[IB_X] - endpoint_one[IB_X])

    #                 distance = math.sqrt(zdiff * zdiff + ydiff * ydiff + xdiff * xdiff)

    #                 if (distance < maximum_distance):
    #                     midpoint = ((endpoint_two[IB_Z] + endpoint_one[IB_Z]) / 2, (endpoint_two[IB_Y] + endpoint_one[IB_Y]) / 2, (endpoint_two[IB_X] + endpoint_one[IB_X]) / 2)
    #                     ground_truth = (seg2gold_mapping[is1] == seg2gold_mapping[is2])
    #                     if not seg2gold_mapping[is1] and not seg2gold_mapping[is2]:
    #                         ground_truth = -1

    #                     midpoints.append(midpoint)
    #                     ground_truths.append(ground_truth)
    #                     labels.append((is1, is2))

    # # create list of candidates
    # positive_candidates = []
    # negative_candidates = []
    # undetermined_candidates = []

    # for iv in range(len(midpoints)):
    #     if ground_truths[iv] == -1: 
    #         undetermined_candidates.append(SkeletonCandidate(labels[iv], midpoints[iv], ground_truths[iv]))
    #     elif ground_truths[iv] == 0: 
    #         negative_candidates.append(SkeletonCandidate(labels[iv], midpoints[iv], ground_truths[iv]))
    #     elif ground_truths[iv] == 1: 
    #         positive_candidates.append(SkeletonCandidate(labels[iv], midpoints[iv], ground_truths[iv]))

    # print 'Number positive edges {}'.format(len(positive_candidates))
    # print 'Number negative edges {}'.format(len(negative_candidates))
    # print 'Number undetermined edges {}'.format(len(undetermined_candidates))

    # # save positive and negative candidates separately
    # positive_filename = 'features/skeleton/{}-{}nm-positive.candidates'.format(prefix, maximum_distance)
    # negative_filename = 'features/skeleton/{}-{}nm-negative.candidates'.format(prefix, maximum_distance)
    # undetermined_filename = 'features/skeleton/{}-{}nm-undetermined.candidates'.format(prefix, maximum_distance)
        
    # SaveCandidates(positive_filename, positive_candidates)
    # SaveCandidates(negative_filename, negative_candidates)
    # SaveCandidates(undetermined_filename, undetermined_candidates)
    # print time.time() - start_time
