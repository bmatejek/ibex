import math
import numpy as np
import random
import struct
from numba import jit
import time

from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.transforms import seg2seg, seg2gold
from ibex.cnns.skeleton.util import SkeletonCandidate
from ibex.data_structures import unionfind
from ibex.evaluation import comparestacks



# save the candidate files for the CNN
def SaveCandidates(output_filename, candidates):
    random.shuffle(candidates)
    ncandidates = len(candidates)

    # write all candidates to the file
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('i', len(candidates)))

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
def FindNeighboringCandidates(segmentation, centroid, candidates, maximum_distance, network_distance, world_res):
    # useful variables
    zres, yres, xres = segmentation.shape
    max_label = np.amax(segmentation) + 1

    # get the radii and label for this centroid
    radii = np.int64((maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X]))
    network_radii = np.int64((network_distance / world_res[IB_Z], network_distance / world_res[IB_Y], network_distance / world_res[IB_X]))
    label = segmentation[centroid[IB_Z],centroid[IB_Y],centroid[IB_X]]

    # iterate through all the pixels close to the centroid
    for iz in range(centroid[IB_Z]-radii[IB_Z], centroid[IB_Z]+radii[IB_Z]+1):
        if iz < 0 or iz > zres - 1: continue
        for iy in range(centroid[IB_Y]-radii[IB_Y], centroid[IB_Y]+radii[IB_Y]+1):
            if iy < 0 or iy > yres - 1: continue
            for ix in range(centroid[IB_X]-radii[IB_X], centroid[IB_X]+radii[IB_X]+1):
                if ix < 0 or ix > xres - 1: continue
                # skip extracellular and locations with the same label
                if not segmentation[iz,iy,ix]: continue
                if segmentation[iz,iy,ix] == label: continue

                # get the distance from the centroid
                distance = math.sqrt((world_res[IB_Z] * (centroid[IB_Z] - iz)) * (world_res[IB_Z] * (centroid[IB_Z] - iz))  + (world_res[IB_Y] * (centroid[IB_Y] - iy)) * (world_res[IB_Y] * (centroid[IB_Y] - iy)) + (world_res[IB_X] * (centroid[IB_X] - ix)) * (world_res[IB_X] * (centroid[IB_X] - ix)))
                if distance > maximum_distance: continue

                # is there already a closer location
                neighbor_label = segmentation[iz,iy,ix]
                candidates.add(neighbor_label)



# generate features for this prefix
def GenerateFeatures(prefix, threshold, maximum_distance, network_distance, endpoint_distance, topology):
    start_time = time.time()

    # read in the relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)
    assert (segmentation.shape == gold.shape)
    zres, yres, xres = segmentation.shape

    # remove small connceted components
    thresholded_segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, threshold=threshold).astype(np.int64)
    max_label = np.amax(segmentation) + 1
    
    # get the grid size and the world resolution
    grid_size = segmentation.shape
    world_res = dataIO.Resolution(prefix)

    # get the radius in grid coordinates
    radii = np.int64((maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X]))
    network_radii = np.int64((network_distance / world_res[IB_Z], network_distance / world_res[IB_Y], network_distance / world_res[IB_X]))
    
    # get all of the skeletons
    if topology: skeletons, endpoints = dataIO.ReadTopologySkeletons(prefix, thresholded_segmentation)
    else: skeletons, _, endpoints = dataIO.ReadSWCSkeletons(prefix, thresholded_segmentation)

    # get the set of all considered pairs
    endpoint_candidates = [set() for _ in range(len(endpoints))]
    for ie, endpoint in enumerate(endpoints):
        # extract the region around this endpoint
        label = endpoint.label
        centroid = endpoint.GridPoint()

        # find the candidates near this endpoint
        candidates = set()
        candidates.add(0)
        FindNeighboringCandidates(thresholded_segmentation, centroid, candidates, maximum_distance, network_distance, world_res)

        for candidate in candidates:
            # skip extracellular
            if not candidate: continue
            endpoint_candidates[ie].add(candidate)

    # get a mapping from the labels to indices in skeletons and endpoints
    label_to_index = [-1 for _ in range(max_label)]
    for ie, skeleton in enumerate(skeletons):
        label_to_index[skeleton.label] = ie 

    # begin pruning the candidates based on the endpoints
    endpoint_pairs = {}

    # find the smallest pair between endpoints
    smallest_distances = (endpoint_distance + 1) * np.ones((max_label, max_label), dtype=np.float32)
    midpoints = np.zeros((max_label, max_label, 3), dtype=np.int32)

    for ie, endpoint in enumerate(endpoints):
        # get the endpoint location
        label = endpoint.label

        # go through all currently considered endpoints
        for neighbor_label in endpoint_candidates[ie]:
            for neighbor_endpoint in skeletons[label_to_index[neighbor_label]].endpoints:
                # get the distance
                deltas = endpoint.WorldPoint(world_res) - neighbor_endpoint.WorldPoint(world_res)
                distance = math.sqrt(deltas[IB_Z] * deltas[IB_Z] + deltas[IB_Y] * deltas[IB_Y] + deltas[IB_X] * deltas[IB_X])

                if distance < smallest_distances[label, neighbor_label]:
                    midpoint = (endpoint.GridPoint() + neighbor_endpoint.GridPoint()) / 2

                    # make sure the bounding box fits
                    valid_location = True
                    for dim in range(NDIMS):
                        if midpoint[dim]-network_radii[dim] < 0: valid_location = False
                        if midpoint[dim]+network_radii[dim] > grid_size[dim]: valid_location = False
                    if not valid_location: continue
            
                    smallest_distances[label,neighbor_label] = distance
                    smallest_distances[neighbor_label,label] = distance

                    # add to the dictionary
                    endpoint_pairs[(label, neighbor_label)] = (endpoint, neighbor_endpoint)
                    endpoint_pairs[(neighbor_label, label)] = (neighbor_endpoint, endpoint)

                    midpoints[label,neighbor_label] = midpoint
                    midpoints[neighbor_label,label] = midpoint

    # create list of candidates
    positive_candidates = []
    negative_candidates = []
    undetermined_candidates = []

    for ie, match in enumerate(endpoint_pairs):
        endpoint_one = endpoint_pairs[match][0]
        endpoint_two = endpoint_pairs[match][1]

        label_one = endpoint_one.label
        label_two = endpoint_two.label

        if (label_one > label_two): continue

        # extract a bounding box around this midpoint
        midz, midy, midx = midpoints[label_one,label_two]

        extracted_segmentation = segmentation[midz-network_radii[IB_Z]:midz+network_radii[IB_Z],midy-network_radii[IB_Y]:midy+network_radii[IB_Y],midx-network_radii[IB_X]:midx+network_radii[IB_X]]
        extracted_gold = gold[midz-network_radii[IB_Z]:midz+network_radii[IB_Z],midy-network_radii[IB_Y]:midy+network_radii[IB_Y],midx-network_radii[IB_X]:midx+network_radii[IB_X]]

        extracted_seg2gold_mapping = seg2gold.Mapping(extracted_segmentation, extracted_gold, low_threshold=0.1, high_threshold=0.40)
        gold_one = extracted_seg2gold_mapping[label_one]
        gold_two = extracted_seg2gold_mapping[label_two]

        ground_truth = (gold_one == gold_two)

        candidate = SkeletonCandidate((label_one, label_two), midpoints[label_one,label_two,:], ground_truth)

        if not extracted_seg2gold_mapping[label_one] or not extracted_seg2gold_mapping[label_two]: undetermined_candidates.append(candidate)
        elif ground_truth: positive_candidates.append(candidate)
        else: negative_candidates.append(candidate)

    # save positive and negative candidates separately
    positive_filename = 'features/skeleton/{}-{}-{}nm-{}nm-{}nm-positive.candidates'.format(prefix, threshold, maximum_distance, endpoint_distance, network_distance)
    negative_filename = 'features/skeleton/{}-{}-{}nm-{}nm-{}nm-negative.candidates'.format(prefix, threshold, maximum_distance, endpoint_distance, network_distance)
    undetermined_filename = 'features/skeleton/{}-{}-{}nm-{}nm-{}nm-undetermined.candidates'.format(prefix, threshold, maximum_distance, endpoint_distance, network_distance)
    
    SaveCandidates(positive_filename, positive_candidates)
    SaveCandidates(negative_filename, negative_candidates)
    SaveCandidates(undetermined_filename, undetermined_candidates)

    print 'Positive candidates: {}'.format(len(positive_candidates))
    print 'Negative candidates: {}'.format(len(negative_candidates))
    print 'Undetermined candidates: {}'.format(len(undetermined_candidates))

    # perform some tests to see how well this method can do
    max_value = np.amax(segmentation) + 1
    union_find = [unionfind.UnionFindElement(iv) for iv in range(max_value)]

    # iterate over all collapsed edges
    for candidate in positive_candidates:
        label_one, label_two = candidate.labels
        unionfind.Union(union_find[label_one], union_find[label_two])
    
    # create a mapping for the labels
    mapping = np.zeros(max_value, dtype=np.int64)
    for iv in range(max_value):
        mapping[iv] = unionfind.Find(union_find[iv]).label
    segmentation = seg2seg.MapLabels(segmentation, mapping)

    comparestacks.CremiEvaluate(segmentation, gold, dilate_ground_truth=1, mask_ground_truth=True, filtersize=0)

