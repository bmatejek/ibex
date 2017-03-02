import argparse
import os
import sys
import numpy as np
import struct

from swc import Skeleton
from scipy.spatial import KDTree

# add parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utilities import dataIO



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('segmentation', type=str, help='filename for segmentation dataset')
    parser.add_argument('gold', type=str, help='filename for gold dataset')
    parser.add_argument('max_distance', type=int, help='maximum distance between two considered endpoints in nanometers')
    parser.add_argument('closest_neighbors', type=int, help='the number of closest neighbors between ')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print progress (default: False)')

    args = parser.parse_args()

    # open the segmentation and gold files
    segmentation = dataIO.ReadH5File(args.segmentation, 'main')
    gold = dataIO.ReadH5File(args.gold, 'stack')

    # get filename prefix
    prefix = args.segmentation.split('/')[1].split('_')[0]

    # read in the meta data
    (zsamp, ysamp, xsamp) = dataIO.ReadMeta(prefix)

    # create an array for all of the skeletons
    skeletons = []

    # read in all skeletons
    nendpoints = 0
    for label in np.unique(segmentation):
        skeleton_filename = 'skeletons/' + prefix + '/' + 'tree_' + str(label) + '.swc'

        # see if this skeleton exists
        if not os.path.isfile(skeleton_filename):
            print 'Unable to read ' + str(skeleton_filename)
            continue

        # read the skeletons
        skeleton = Skeleton(prefix, label)
        nendpoints += len(skeleton.endpoints)

        # add to the list of skeletons
        skeletons.append(skeleton)

    # create a data array for the kdtree and mapping from endpoint to label
    data = np.zeros((nendpoints, 3), dtype=np.float32)
    endpoint_labels = np.zeros(nendpoints, dtype=np.uint64)

    index = 0
    for skeleton in skeletons:
        for endpoint in skeleton.endpoints:

            # set the x, y, and z coordinates for the skeleton
            data[index,0] = xsamp * endpoint.x
            data[index,1] = ysamp * endpoint.y
            data[index,2] = zsamp * endpoint.z

            # update the label mapping
            endpoint_labels[index] = skeleton.label

            index += 1

    # create a KDTree
    kdtree = KDTree(data)

    # find all pairs of neighbors within max_distance
    close_pairs = kdtree.query_ball_tree(kdtree, args.max_distance)


    # create an array of all boundary examples
    potential_merges = []

    # get all pairs of endpoints that should be considered
    for index_one, pairs in enumerate(close_pairs):
        for index_two in pairs:
            if (index_one > index_two): continue
            if endpoint_labels[index_one] == endpoint_labels[index_two]: continue

            potential_merges.append((index_one, index_two))

    output_filename = 'skeletons/' + prefix + '_merge_candidates.merge'

    with open(output_filename, 'wb') as fd:
        # write the number of potential merges
        fd.write(struct.pack('Q', len(potential_merges)))

        # find the center for all of the boundary examples
        for pair in potential_merges:
            index_one = pair[0]
            index_two = pair[1]

            # get the location for the two skeleton endpoints
            position_one = data[index_one,:]
            position_two = data[index_two,:]

            # find the middle point for this merge
            mid_point = (position_one + position_two) / 2

            # create a string of relevant information
            fd.write(struct.pack('QQQQddd', index_one, index_two, endpoint_labels[index_one], endpoint_labels[index_two], mid_point[0] / xsamp, mid_point[1] / ysamp, mid_point[2] / zsamp))


if __name__ == '__main__':
    main()    
