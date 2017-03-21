import argparse
import numpy as np
import sys
import os
import time
from keras.models import Sequential, load_model

# add parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utilities import dataIO
from skeleton_classifier import make_window, ReadMergeFilename
from evaluation import classification


def data_generator(args, prefix):
    # read in h5 file
    filename = 'rhoana/' + prefix + '_rhoana.h5'
    segmentation = dataIO.ReadH5File(filename, 'main')

    # read in potential merge locations
    merge_filename = 'skeletons/' + prefix + '_merge_candidates.merge'
    merge_candidates, npositives, nnegatives = ReadMergeFilename(merge_filename)

    num_locations = len(merge_candidates)

    examples = np.zeros((num_locations, args.window_width, args.window_width, args.window_width, 3))
    labels = np.zeros(num_locations)

    for index in range(num_locations):
        # get this merge candidate
        merge_candidate = merge_candidates[index]

        window = make_window(segmentation, merge_candidate.label_one, merge_candidate.label_two, merge_candidate.x, merge_candidate.y, merge_candidate.z, args.window_width)

        examples[index,:,:,:,:] = window
        labels[index] = merge_candidate.ground_truth

    return (examples, labels)


def main():
    # parse the arguments
    parser = argparse.ArgumentParser(description='Train and output a classifier for skeletons')
    parser.add_argument('prefix', help='Prefix for the dataset')
    parser.add_argument('model', help='The classifier trained in skeleton_train.py')
    parser.add_argument('--window_width', default=51, type=int, help='Width of window in each dimension')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='print verbosity')
    args = parser.parse_args()

    # load the model
    model = load_model(args.model)

    # read in all of the data

    (examples, labels) = data_generator(args, args.prefix)

    probabilities = model.predict_proba(examples, verbose=1)
    nexamples = probabilities.shape[0]

    # create the predictions for precision and recall
    predictions = np.zeros(nexamples, dtype=np.uint8)
    for ie in range(nexamples):
        if probabilities[ie,1] > probabilities[ie,0]:
            predictions[ie] = 1
        else:
            predictions[ie] = 0

    classification.PrecisionAndRecall(labels, predictions)

if __name__ == '__main__':
    main()