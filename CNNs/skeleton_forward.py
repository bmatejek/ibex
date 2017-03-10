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


def data_generator(args, prefix):
    # read in h5 file
    filename = 'rhoana/' + prefix + '_rhoana.h5'
    segmentation = dataIO.ReadH5File(filename, 'main')

    # read in potential merge locations
    merge_filename = 'skeletons/' + prefix + '_merge_candidates.merge'
    merge_candidates = ReadMergeFilename(merge_filename)

    num_locations = len(merge_candidates)

    examples = np.zeros((num_locations, args.window_width, args.window_width, args.window_width, 1))
    labels = np.zeros((num_locations, 2))

    for index in range(num_locations):
        # get this merge candidate
        merge_candidate = merge_candidates[index]

        window, label = make_window(segmentation, merge_candidate.label_one, merge_candidate.label_two, merge_candidate.x, merge_candidate.y, merge_candidate.z, merge_candidate.ground_truth, args.window_width)

        examples[index,:,:,:,0] = window
        labels[index,:] = label

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

    print 'Reading in data...',
    start_time = time.time()
    (examples, labels) = data_generator(args, args.prefix)
    print 'done in %0.2f seconds' % (time.time() - start_time)
    print 'Generating predictions...'
    predictions = model.predict(examples, verbose=1)

    # get the precision and recall
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for ie in range(len(predictions)):
        label = bool(1 - labels[ie,0])
        prediction = (predictions[ie,1] > predictions[ie,0])
        if label and prediction:
            true_positives += 1
        elif not label and prediction:
            false_positives += 1
        elif label and not prediction:
            false_negatives += 1
        else:
            true_negatives += 1

    print 'TP: ' + str(true_positives)
    print 'FP: ' + str(false_positives)
    print 'FN: ' + str(false_negatives)
    print 'TN: ' + str(true_negatives)

    print 'Precision: ' + str(float(true_positives) / float(true_positives + false_positives))
    print 'Recall: ' + str(float(true_positives) / float(true_positives + false_negatives))
    print 'Accuracy: ' + str(float(true_positives + true_negatives) / float(true_positives + false_positives + false_negatives + true_negatives))

if __name__ == '__main__':
    main()