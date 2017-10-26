import numpy as np
import random
import struct
import math
from numba import jit

from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.transforms import seg2seg, seg2gold
from ibex.cnns.skeleton.util import SkeletonCandidate
from ibex.data_structures import unionfind
from PixelPred2Seg import comparestacks




# save the candidate files for the CNN
def SaveCandidates(output_filename, positive_candidates, negative_candidates, inference=False, undetermined_candidates=None):
    if not undetermined_candidates == None:
        candidates = undetermined_candidates
        random.shuffle(candidates)
    elif inference:
        # concatenate the two lists
        candidates = positive_candidates + negative_candidates
        random.shuffle(candidates)
    else:
        # randomly shuffle the arrays
        random.shuffle(positive_candidates)
        random.shuffle(negative_candidates)

        # get the minimum length of the two candidates - train in pairs
        npoints = max(len(positive_candidates), len(negative_candidates))
        positive_index = 0
        negative_index = 0

        # train in pairs, duplicate when needed
        candidates = []
        for _ in range(npoints):
            candidates.append(positive_candidates[positive_index])
            candidates.append(negative_candidates[negative_index])

            # increment the indices
            positive_index += 1
            negative_index += 1

            # handle dimension mismatch by reseting index and reshuffling array
            if positive_index >= len(positive_candidates): 
                positive_index = 0
                random.shuffle(positive_candidates)
            if negative_index >= len(negative_candidates): 
                negative_index = 0
                random.shuffle(negative_candidates)
            
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

            # get the ground truth for this candidate
            ground_truth = candidate.ground_truth

            # write this candidate to the evaluation candidate list
            fd.write(struct.pack('QQQQQQ', label_one, label_two, position[IB_Z], position[IB_Y], position[IB_X], ground_truth))



@jit(nopython=True)
# extract the neighbors from this region
def ExtractNeighbors(segmentation, label, smallest_distance, nearest_point, network_radii, world_res, grid_size, centroid):
    zres, yres, xres = segmentation.shape
    
    # iterate over the entire segment
    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                # get the label for this neighbor
                neighbor_label = segmentation[iz,iy,ix]
                if neighbor_label == label: continue

                # get dostamce from center
                distance = (world_res[IB_Z] * iz) ^ 2 + (world_res[IB_Y] * iy) ^ 2 + (world_res[IB_X] * ix) ^ 2
                if distance < smallest_distance[neighbor_label]:
                    
                    # get what would be the midpoint of this bounding box in world coordinates
                    midpoint = (iz - zres / 2 + centroid[IB_Z], iy - yres / 2 + centroid[IB_Y], ix - xres / 2 + centroid[IB_X])
                    # make sure the entire bounding box is enclosed
                    valid_location = True
                    for dim in range(NDIMS):
                        if (midpoint[dim] - network_radii[dim] < 0): valid_location = False
                        if (midpoint[dim] + network_radii[dim] > grid_size[dim]): valid_location = False
                    if not valid_location: continue

                    smallest_distance[neighbor_label] = distance
                    nearest_point[neighbor_label,:] = (midpoint[IB_Z], midpoint[IB_Y], midpoint[IB_X])



@jit(nopython=True)
def CollectPoints(segmentation, label_one, label_two):
    # get the shape of the segmentation
    zres, yres, xres = segmentation.shape
    zstep, ystep, xstep = (2, 10, 10)

    # npoints = 0
    # for iz in range(0, zres, zstep):
    #     for iy in range(0, yres, ystep):
    #         for ix in range(0, xres, xstep):
    #             if segmentation[iz,iy,ix] == label_one or segmentation[iz,iy,ix] == label_two:
    #                 npoints += 1

    nlabel_one_points = 0
    nlabel_two_points = 0

    onex, oney, onez = (0, 0, 0)
    twox, twoy, twoz = (0, 0, 0)

    for iz in range(0, zres, 2):
        for iy in range(0, yres, 10):
            for ix in range(0, xres, 10):
                if segmentation[iz,iy,ix] == label_one: 
                    nlabel_one_points += 1
                    onex += ix
                    oney += iy
                    onez += iz
                if segmentation[iz,iy,ix] == label_two: 
                    nlabel_two_points += 1
                    twox += ix
                    twoy += iy
                    twoz += iz

    one_centerx, one_centery, one_centerz = (onex / float(nlabel_one_points), oney / float(nlabel_one_points), onez / float(nlabel_one_points))
    two_centerx, two_centery, two_centerz = (twox / float(nlabel_one_points), twoy / float(nlabel_two_points), twoz / float(nlabel_two_points))
    stddev_one = 0.0
    stddev_two = 0.0

    label_one_points = np.zeros((nlabel_one_points, 3), dtype=np.float32)
    label_two_points = np.zeros((nlabel_two_points, 3), dtype=np.float32)

    one_seen = 0
    two_seen = 0

    for iz in range(0, zres, 2):
        for iy in range(0, yres, 10):
            for ix in range(0, xres, 10):
                if segmentation[iz,iy,ix] == label_one:
                    label_one_points[one_seen,:] = (iz, iy, ix)
                    one_seen += 1
                    stddev_one += (iz - one_centerz) * (iz - one_centerz) + (iy - one_centery) * (iy - one_centery) + (ix - one_centerx) * (ix - one_centerx)
                if segmentation[iz,iy,ix] == label_two:
                    label_two_points[two_seen,:] = (iz, iy, ix)
                    stddev_two += (iz - two_centerz) * (iz - two_centerz) + (iy - two_centery) * (iy - two_centery) + (ix - two_centerx) * (ix - two_centerx)
                    two_seen += 1

    stddev_one = math.sqrt(stddev_one / (nlabel_one_points - 1))
    stddev_two = math.sqrt(stddev_two / (nlabel_two_points - 1))


    for iv in range(nlabel_one_points):
        label_one_points[iv,0] = (label_one_points[iv,0] - one_centerz) / stddev_one
        label_one_points[iv,1] = (label_one_points[iv,1] - one_centery) / stddev_one
        label_one_points[iv,2] = (label_one_points[iv,2] - one_centerx) / stddev_one
    for iv in range(nlabel_two_points):
        label_two_points[iv,0] = (label_two_points[iv,0] - two_centerz) / stddev_two
        label_two_points[iv,1] = (label_two_points[iv,1] - two_centery) / stddev_two
        label_two_points[iv,2] = (label_two_points[iv,2] - two_centerx) / stddev_two


    return label_one_points, label_two_points


# generate the features given this threshold and distance
def GenerateFeatures(prefix, threshold, maximum_distance, network_distance):
    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)
    assert (segmentation.shape == gold.shape)

    # remove small connected components
    segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, threshold=threshold)
    max_label = np.amax(segmentation) + np.uint64(1)

    # get the grid size and the world resolution
    grid_size = segmentation.shape
    world_res = dataIO.Resolution(prefix)

    # get the radius in grid coordinates
    radii = np.uint64((maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X]))
    network_radii = np.uint64((network_distance / world_res[IB_Z], network_distance / world_res[IB_Y], network_distance / world_res[IB_X]))



    # create the list of candidates
    positive_candidates = []
    negative_candidates = []
    undetermined_candidates = []



    # read in the skeletons, ignore the joints
    skeletons, _, endpoints = dataIO.ReadSkeletons(prefix, segmentation)

    with open('{}.pca'.format(prefix), 'w') as fd:

        # go through every skeleton
        for skeleton in skeletons:
            label_one = skeleton.label

            # keep track of the best distances from other labels to this
            max_distance = (world_res[IB_Z] * segmentation.shape[IB_Z]) ^ 2 + (world_res[IB_Y] * segmentation.shape[IB_Y]) ^ 2 + (world_res[IB_X] * segmentation.shape[IB_X]) ^ 2
            smallest_distance = np.ones(max_label, dtype=np.uint64) * max_distance
            nearest_point = np.zeros((max_label, 3), dtype=np.uint64)
            
            # iterate through every endpoint
            for endpoint in skeleton.endpoints:
                # the center of the region of interest
                centroid = endpoint.GridPoint()

                # get all labels in this bounding box
                segment = segmentation[centroid[IB_Z]-radii[IB_Z]:centroid[IB_Z]+radii[IB_Z],centroid[IB_Y]-radii[IB_Y]:centroid[IB_Y]+radii[IB_Y],centroid[IB_X]-radii[IB_X]:centroid[IB_X]+radii[IB_X]]
                ExtractNeighbors(segment, label_one, smallest_distance, nearest_point, network_radii, world_res, grid_size, centroid)

            # only consider pairs where this skeleton has a smaller label
            for label_two in range(label_one + np.uint64(1), max_label):
                if smallest_distance[label_two] == max_distance: continue

                # get the midpoint
                midpoint = nearest_point[label_two]
                assert (not midpoint[IB_Z] == 0 and not midpoint[IB_Y] == 0 and not midpoint[IB_X] == 0)
                mins = midpoint - network_radii
                maxs = midpoint + network_radii

                # extra ct a small window around this midpoing
                extracted_segmentation = segmentation[mins[IB_Z]:maxs[IB_Z],mins[IB_Y]:maxs[IB_Y],mins[IB_X]:maxs[IB_X]]
                extracted_gold = gold[mins[IB_Z]:maxs[IB_Z],mins[IB_Y]:maxs[IB_Y],mins[IB_X]:maxs[IB_X]]

                # create the gold to segmentation mapping
                seg2gold_mapping = seg2gold.Mapping(extracted_segmentation, extracted_gold)
                ground_truth = (seg2gold_mapping[label_one] == seg2gold_mapping[label_two])

                # create the candidate
                candidate = SkeletonCandidate((label_one, label_two), midpoint, ground_truth)
                if not seg2gold_mapping[label_one] or not seg2gold_mapping[label_two]:
                    undetermined_candidates.append(candidate)
                    continue

                if ground_truth: positive_candidates.append(candidate)
                else: negative_candidates.append(candidate)

                # get the PCA analysis
                label_one_points, label_two_points = CollectPoints(extracted_segmentation, label_one, label_two)

                _, _, Vone = np.linalg.svd(label_one_points)
                _, _, Vtwo = np.linalg.svd(label_two_points)

                # get priniciple eigenvectors
                e_one = Vone[0,:] / np.linalg.norm(Vone[0,:])
                e_two = Vtwo[0,:] / np.linalg.norm(Vtwo[0,:])
                
                fd.write('{},{},{},{}\n'.format(label_one, label_two, ground_truth, math.acos(e_one.dot(e_two))))
                print '{},{},{},{}'.format(label_one, label_two, ground_truth, math.acos(e_one.dot(e_two)))




    # print statistics
    print 'Results for {}, threshold {}, maximum distance {}, network distance {}:'.format(prefix, threshold, maximum_distance, network_distance)
    print '  Positive examples: {}'.format(len(positive_candidates))
    print '  Negative examples: {}'.format(len(negative_candidates))
    print '  Ratio: {}'.format(len(negative_candidates) / float(len(positive_candidates)))

    # # save the files
    # train_filename = 'features/skeleton/{}-{}-{}nm-{}nm-learning.candidates'.format(prefix, threshold, maximum_distance, network_distance)
    # forward_filename = 'features/skeleton/{}-{}-{}nm-{}nm-inference.candidates'.format(prefix, threshold, maximum_distance, network_distance)
    # undetermined_filename = 'features/skeleton/{}-{}-{}nm-{}nm-undetermined.candidates'.format(prefix, threshold, maximum_distance, network_distance)

    # SaveCandidates(train_filename, positive_candidates, negative_candidates, inference=False)
    # SaveCandidates(forward_filename, positive_candidates, negative_candidates, inference=True)
    # SaveCandidates(undetermined_filename, positive_candidates, negative_candidates, undetermined_candidates=undetermined_candidates)



    # # perform some tests to see how well this method can do
    # max_value = np.amax(segmentation) + np.uint64(1)
    # union_find = [unionfind.UnionFindElement(iv) for iv in range(max_value)]

    # # iterate over all collapsed edges
    # for candidate in positive_candidates:
    #     label_one, label_two = candidate.labels
    #     unionfind.Union(union_find[label_one], union_find[label_two])

    # # create a mapping for the labels
    # mapping = np.zeros(max_value, dtype=np.uint64)
    # for iv in range(max_value):
    #     mapping[iv] = unionfind.Find(union_find[iv]).label
    # opt_segmentation = seg2seg.MapLabels(segmentation, mapping)
    # comparestacks.Evaluate(opt_segmentation, gold)
