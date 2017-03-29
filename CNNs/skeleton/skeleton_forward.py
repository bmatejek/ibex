import argparse
import numpy as np
import sys
import os
import time
from keras.models import Model, Sequential, model_from_json
import struct

# add parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utilities import dataIO
from skeleton_classifier import make_window, ReadMergeFilename
from evaluation import classification
from transforms import seg2seg
from keras import backend as K
from skeleton_train import weighted_mse, maybe_print

def ReadCandidates(args, prefix):
    # read in potential merge locations
    # TODO remove hardcoding
    merge_filename = 'skeletons/' + prefix + '_merge_candidates_forward_400nm.merge'
    merge_candidates, _, _, _ = ReadMergeFilename(merge_filename)

    num_locations = len(merge_candidates)
    labels = np.zeros(num_locations)

    for index in range(num_locations):
        # get this merge candidate
        merge_candidate = merge_candidates[index]

        labels[index] = merge_candidate.ground_truth

    return merge_candidates, labels

def data_generator(args, prefix):
    # read in h5 file
    filename = 'rhoana/' + prefix + '_rhoana.h5'
    segmentation = dataIO.ReadH5File(filename, 'main')

    # read in potential merge locations
    # TODO remove hardcoding
    merge_filename = 'skeletons/' + prefix + '_merge_candidates_forward_400nm.merge'
    merge_candidates, npositives, nnegatives, radii = ReadMergeFilename(merge_filename)

    index = 0

    while True:
        if (index % 100 == 0): 
            print index
        # get this merge candidate
        if index >= len(merge_candidates):
            print index
            merge_candidate = merge_candidates[0]
        else:
            merge_candidate = merge_candidates[index]

        # get the labels for this candidate
        label_one = merge_candidate.label_one
        label_two = merge_candidate.label_two

        # get the position for this candidate
        xposition = merge_candidate.x
        yposition = merge_candidate.y
        zposition = merge_candidate.z

        window = make_window(segmentation, label_one, label_two, xposition, yposition, zposition, radii, args.window_width)

        example = np.zeros((1, args.window_width, args.window_width, args.window_width, 1))
        example[0,:,:,:,:] = window

        index += 1

        yield example


def GenerateMultiCutInput(args, segmentation):
    # create the output for multi-cut algorithm
    filename = 'rhoana/' + args.prefix + '_rhoana.h5'
    segmentation = dataIO.ReadH5File(filename, 'main')

    # get the mapping to a smaller set of vertices
    forward_mapping, reverse_mapping = seg2seg.ReduceLabels(segmentation)

    # create multi-cut file
    multicut_filename = 'multicut/' + args.prefix + '_skeleton_400nm.graph'

    # open a file to write multi-cut information
    with open(multicut_filename, 'wb') as fd:
        # write the number of vertices and the number of edges
        fd.write(struct.pack('QQ', reverse_mapping.size, len(candidates)))

        # for every merge candidate, determine the weight of the edge
        for ie  in range(len(candidates)):
            candidate = candidates[ie]

            # get the probability of merge from neural network
            probability = probabilities[ie,1]

            # get the labels for these two candidates
            label_one = candidate.label_one
            label_two = candidate.label_two

            # get the new label
            reduced_label_one = forward_mapping[label_one]
            reduced_label_two = forward_mapping[label_two]

            # write the label for both segments and the probability of merge from neural network
            fd.write(struct.pack('QQd', reduced_label_one, reduced_label_two, probability))



def main():
    # parse the arguments
    parser = argparse.ArgumentParser(description='Train and output a classifier for skeletons')
    parser.add_argument('prefix', help='Prefix for the dataset')
    parser.add_argument('model', help='The classifier trained in skeleton_train.py')
    parser.add_argument('--window_width', default=51, type=int, help='Width of window in each dimension')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='print verbosity')
    args = parser.parse_args()

    # load the model
    #model = load_model(args.model)
    model = model_from_json(open(args.model.replace('h5', 'json'), 'r').read())
    model.load_weights(args.model)

    # get the candidate locations
    candidates, labels = ReadCandidates(args, args.prefix)

    print len(candidates)

    # generate probabilities and predictions
    probabilities = model.predict_generator(data_generator(args, args.prefix), len(candidates))
    predictions = classification.prob2pred(probabilities)

    # output the accuracy of this network
    classification.PrecisionAndRecall(labels, predictions)



if __name__ == '__main__':
    main()