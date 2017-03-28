import argparse
import os
import sys
import numpy as np
import struct
import random

from swc import Skeleton
from scipy.spatial import KDTree

# add parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utilities import dataIO, seg2gold

def ReadSkeletons(prefix, segmentation):
    # create an array for all of the skeletons
    skeletons = []

    # read in all skeletons
    npoints = 0
    for label in np.unique(segmentation):
        skeleton_filename = 'skeletons/' + prefix + '/' + 'tree_' + str(label) + '.swc'

        # see if this skeleton exists
        if not os.path.isfile(skeleton_filename):
            continue

        # read the skeletons
        skeleton = Skeleton(prefix, label)
        npoints += len(skeleton.endpoints)

        # add to the list of skeletons
        skeletons.append(skeleton)

    return skeletons, npoints



def GenerateKDTree(skeletons, npoints, world_res):
    # create a locations array for the kdtree and mapping from endpoint to label
    locations = np.zeros((npoints, 3), dtype=np.float32)
    point_labels = np.zeros(npoints, dtype=np.uint64)

    # get the sampling resolution for each dimension
    (zsamp, ysamp, xsamp) = world_res

    index = 0
    for skeleton in skeletons:
        for endpoint in skeleton.endpoints:

            # set the x, y, and z coordinates for the skeleton
            locations[index,0] = xsamp * endpoint.x
            locations[index,1] = ysamp * endpoint.y
            locations[index,2] = zsamp * endpoint.z

            # update the label mapping
            point_labels[index] = skeleton.label

            index += 1

    # create a KDTree
    kdtree = KDTree(locations)

    return locations, kdtree, point_labels



def GenerateMergeLocations(endpoint_pairs, point_labels):
    # create an array of all boundary examples
    merge_locations = []

    # get all pairs of endpoints that should be considered
    for index_one, pairs in enumerate(endpoint_pairs):
        for index_two in pairs:
            # avoid double counting
            if (index_one > index_two): continue

            # if these two endpoints belong to the same segment they can't be merged
            if point_labels[index_one] == point_labels[index_two]: continue

            merge_locations.append((index_one, index_two))

    return merge_locations



def SaveCNNFile(prefix, pairs, locations, labels, seg2gold_mapping, dim_size, world_res, max_distance, forward=False):
    if (forward): filename = 'skeletons/' + prefix + '_merge_candidates_forward_' + str(max_distance) + 'nm.merge'
    else: filename = 'skeletons/' + prefix + '_merge_candidates_train_' + str(max_distance) + 'nm.merge'

    # get the size of each dimension
    (zres, yres, xres) = dim_size

    # get the sampling resolution for each dimension
    (zsamp, ysamp, xsamp) = world_res

    # get the radius for the bounding box in grid coordinates
    (zradius, yradius, xradius) = (max_distance / zsamp, max_distance / ysamp, max_distance / xsamp)

    # keep track of the number of positive and negative locations
    npositives = 0
    nnegatives = 0

    # write to a binary file
    with open(filename, 'wb') as fd:
        # write an empty header
        fd.write(struct.pack('QQ', 0, 0))
        fd.write(struct.pack('QQQ', 0, 0, 0))

        # iterate through all pairs
        for pair in pairs:
            # get the indices of the endpoints
            index_one = pair[0]
            index_two = pair[1]

            # get the positions of the endpoints
            position_one = locations[index_one,:]
            position_two = locations[index_two,:]

            # find the middle point between these locations
            midpoint = (position_one + position_two) / 2

            # get the downsampled location
            xpoint = long(midpoint[0] / xsamp)
            ypoint = long(midpoint[1] / ysamp)
            zpoint = long(midpoint[2] / zsamp)

            # get the label for both points
            label_one = labels[index_one]
            label_two = labels[index_two]

            # get the ground truth
            ground_truth = (seg2gold_mapping[label_one] == seg2gold_mapping[label_two])

            # only include locations that do not extend past the boundary
            if (xpoint - xradius < 0 or xpoint + xradius > xres - 1): continue
            if (ypoint - yradius < 0 or ypoint + yradius > yres - 1): continue
            if (zpoint - zradius < 0 or zpoint + zradius > zres - 1): continue

            # add in augmentation here for training
            for aug_iter in range(1):
                if ground_truth: npositives += 1
                else: nnegatives += 1

                # output the labels corresponding to this segment
                fd.write(struct.pack('QQ', label_one, label_two))
                # write the midpoint coordinates in the grid coordiante system
                fd.write(struct.pack('QQQ', xpoint, ypoint, zpoint))
                # write the ground truth for this merge pair
                fd.write(struct.pack('B', ground_truth))
                # write a final variable corresponding to a rotation in training
                fd.write(struct.pack('B', aug_iter))

                # if this is for a CNN forward pass file skip augmentation
                if forward: break


        # rewrite the header with the number of examples
        fd.seek(0)
        fd.write(struct.pack('QQ', npositives, nnegatives))
        fd.write(struct.pack('QQQ', xradius, yradius, zradius))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('segmentation', type=str, help='filename for segmentation dataset')
    parser.add_argument('gold', type=str, help='filename for gold dataset')
    parser.add_argument('max_distance', type=int, help='maximum distance between two considered endpoints in nanometers')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print progress (default: False)')

    args = parser.parse_args()

    # open the segmentation and gold files
    segmentation = dataIO.ReadH5File(args.segmentation, 'main')
    gold = dataIO.ReadH5File(args.gold, 'stack')
    assert (segmentation.shape == gold.shape)

    # get filename prefix
    prefix = args.segmentation.split('/')[1].split('_')[0]

    # read in the meta data which includes the sampling resolution in nanometers
    dim_size = segmentation.shape
    world_res = dataIO.ReadMeta(prefix)

    # read in the skeleton and number of endpoints
    skeletons, npoints = ReadSkeletons(prefix, segmentation)

    # create a kd tree
    locations, kdtree, point_labels = GenerateKDTree(skeletons, npoints, world_res)

    # find all pairs of neighbors within max_distance
    endpoint_pairs = kdtree.query_ball_tree(kdtree, args.max_distance)

    # find all of the merge locations
    merge_locations = GenerateMergeLocations(endpoint_pairs, point_labels)

    # randomize the locations
    random.shuffle(merge_locations)

    # create a mapping from segmentation to gold
    seg2gold_mapping = seg2gold.seg2gold(segmentation, gold)

    # create training file for training data
    SaveCNNFile(prefix, merge_locations, locations, point_labels, seg2gold_mapping, dim_size, world_res, args.max_distance)

    # create file for forward pass on data
    SaveCNNFile(prefix, merge_locations, locations, point_labels, seg2gold_mapping, dim_size, world_res, args.max_distance, forward=True)

if __name__ == '__main__':
    main()    
