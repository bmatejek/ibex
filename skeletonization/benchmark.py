import struct

import numpy as np

from ibex.data_structures import skeleton
from ibex.utilities import dataIO



# read ground truth
def SkeletonGroundTruth(prefix, max_label, cutoff=500):
    # read the mapping from example number to label
    example_filename = 'benchmarks/skeleton/{}-skeleton-benchmark-examples.bin'.format(prefix)
    with open(example_filename, 'rb') as fd:
        input_cutoff, = struct.unpack('q', fd.read(8))
        assert (cutoff == input_cutoff)

        example_to_label = np.zeros(input_cutoff, dtype=np.int64)
        for ie in range(input_cutoff):
            example_to_label[ie], = struct.unpack('q', fd.read(8))

    skeletons = [None] * max_label

    # read the manually annotated ground truth
    directory = 'benchmarks/skeleton/{}'.format(prefix)
    for iv in range(cutoff):
        filename = '{}/example-{}.pts'.format(directory, iv)

        # create instance variables for future Skeleton object
        label = example_to_label[iv]
        joints = []
        endpoints = []

        with open(filename, 'rb') as fd:
            npoints, = struct.unpack('q', fd.read(8))

            for ip in range(npoints):
                iz, iy, ix, = struct.unpack('qqq', fd.read(24))
                endpoints.append((iz, iy, ix))

        skeletons[label] = skeleton.Skeleton(label, joints, endpoints)

    return skeletons



# evaluate the methods
def EvaluateSkeletons(prefix, cutoff=500, resolution=(100, 100, 100)):
    # read all of the skeletons
    medial_skeletons = dataIO.ReadSkeletons(prefix, skeleton_algorithm='medial-axis', benchmark=True)
    teaser_skeletons = dataIO.ReadSkeletons(prefix, skeleton_algorithm='teaser',  benchmark=True)
    thinning_skeletons = dataIO.ReadSkeletons(prefix, skeleton_algorithm='thinning', benchmark=True)

    ground_truth = SkeletonGroundTruth(prefix, cutoff)



# create a list of endpoints to consider to finetune ground truth
def InteractiveEditExamples(prefix, cutoff=500, resolution=(100, 100, 100)):
    # read in the the thinning skeleton
    thinning_skeletons = dataIO.ReadSkeletons(prefix, skeleton_algorithm='thinning', benchmark=True).skeletons
    max_label = len(thinning_skeletons)

    # read the ground truth skeletons
    ground_truth_skeletons = SkeletonGroundTruth(prefix, max_label, cutoff)

    # go through all labels
    for label in range(max_label):
        if ground_truth_skeletons[label] == None: continue

        # get the topological thinning and ground truth skeleton
        thinning_skeleton = thinning_skeletons[label].Endpoints2Array()
        ground_truth_skeleton = ground_truth_skeletons[label].Endpoints2Array()

        print label
        print thinning_skeleton.shape
        print ground_truth_skeleton.shape



# find skeleton benchmark information
def GenerateExamples(prefix, cutoff=500):
    gold = dataIO.ReadGoldData(prefix)
    labels, counts = np.unique(gold, return_counts=True)

    filename = 'benchmarks/skeleton/{}-skeleton-benchmark-examples.bin'.format(prefix)
    with open(filename, 'wb') as fd:
        fd.write(struct.pack('q', cutoff))
        if labels[0] == 0: cutoff += 1
        for ie, (count, label) in enumerate(sorted(zip(counts, labels), reverse=True)):
            if not label: continue
            # don't include more than cutoff examples
            if ie == cutoff: break
            fd.write(struct.pack('q', label))