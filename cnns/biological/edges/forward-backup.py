import os
import numpy as np
import sys
import struct
import time

from keras.models import model_from_json

from ibex.utilities import dataIO
from ibex.utilities.constants import *
from ibex.cnns.biological.util import AugmentFeature
from ibex.evaluation.classification import Prob2Pred, PrecisionAndRecall



# generator for the inference of the neural network
def EdgeGenerator(examples, width):
    index = 0

    start_time = time.time()

    while True:
        if index and not (index % 1000):
            print '{}/{} in {:0.2f} seconds'.format(index, examples.shape[0], time.time() - start_time)
            start_time = time.time()
        # prevent overflow of the queue (these examples will not go through)
        if index == examples.shape[0]: index = 0

        # augment the feature
        example = AugmentFeature(examples[index], width)

        # update the index
        index += 1
        
        yield example



def CollectExamples(prefix, width, radius, subset):
    # get the parent directory with all of the featuers
    parent_directory = 'features/biological/edges-{}nm-{}x{}x{}'.format(radius, width[IB_Z+1], width[IB_Y+1], width[IB_X+1])
    
    positive_filename = '{}/{}/positives/{}-examples.h5'.format(parent_directory, subset, prefix)
    positive_examples = dataIO.ReadH5File(positive_filename, 'main')

    negative_filename = '{}/{}/negatives/{}-examples.h5'.format(parent_directory, subset, prefix)
    negative_examples = dataIO.ReadH5File(negative_filename, 'main')
    
    unknowns_filename = '{}/{}/unknowns/{}-examples.h5'.format(parent_directory, subset, prefix)
    unknowns_example = dataIO.ReadH5File(unknowns_filename, 'main')

    # concatenate all of the examples together
    examples = np.concatenate((positive_examples, negative_examples, unknowns_example), axis=0)
    
    # add in information needed for forward inference [regions masked out for training and validation]
    forward_positive_filename = '{}/forward/positives/{}-examples.h5'.format(parent_directory, prefix)
    if os.path.exists(forward_positive_filename):
        forward_positive_examples = dataIO.ReadH5File(forward_positive_filename, 'main')
        examples = np.concatenate((examples, forward_positive_examples), axis=0)

    forward_negative_filename = '{}/forward/negatives/{}-examples.h5'.format(parent_directory, prefix)
    if os.path.exists(forward_negative_filename):
        forward_negative_examples = dataIO.ReadH5File(forward_negative_filename, 'main')
        examples = np.concatenate((examples, forward_negative_examples), axis=0)

    forward_unknowns_filename = '{}/forward/unknowns/{}-examples.h5'.format(parent_directory, prefix)
    if os.path.exists(forward_unknowns_filename):
        forward_unknowns_examples = dataIO.ReadH5File(forward_unknowns_filename, 'main')
        examples = np.concatenate((examples, forward_unknowns_examples), axis=0)

    return examples, positive_examples.shape[0], negative_examples.shape[0]



def CollectEdges(prefix, width, radius, subset):
    # get the parent directory with all of the features
    parent_directory = 'features/biological/edges-{}nm-{}x{}x{}'.format(radius, width[IB_Z+1], width[IB_Y+1], width[IB_X+1])

    examples = []

    positive_filename = '{}/{}/positives/{}.examples'.format(parent_directory, subset, prefix)
    with open(positive_filename, 'rb') as fd:
        nexamples, = struct.unpack('q', fd.read(8))
        for _ in range(nexamples):
            _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
            examples.append((label_one, label_two))

    negative_filename = '{}/{}/negatives/{}.examples'.format(parent_directory, subset, prefix)
    with open(negative_filename, 'rb') as fd:
        nexamples, = struct.unpack('q', fd.read(8))
        for _ in range(nexamples):
            _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
            examples.append((label_one, label_two))

    unknowns_filename = '{}/{}/unknowns/{}.examples'.format(parent_directory, subset, prefix)
    with open(unknowns_filename, 'rb') as fd:
        nexamples, = struct.unpack('q', fd.read(8))
        for _ in range(nexamples):
            _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
            examples.append((label_one, label_two))


    # add in information needed for forward inference [regions masked out for training and validation]
    forward_positive_filename = '{}/forward/positives/{}.examples'.format(parent_directory, prefix)
    if os.path.exists(forward_positive_filename):
        with open(forward_positive_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
                examples.append((label_one, label_two))

    forward_negative_filename = '{}/forward/negatives/{}.examples'.format(parent_directory, prefix)
    if os.path.exists(forward_negative_filename):
        with open(forward_negative_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
                examples.append((label_one, label_two))

    forward_unknowns_filename = '{}/forward/unknowns/{}.examples'.format(parent_directory, prefix)
    if os.path.exists(forward_unknowns_filename):
        with open(forward_unknowns_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
                examples.append((label_one, label_two))

    return examples



def Forward(prefix, model_prefix, width, radius, subset, evaluate=False):
    # read in the trained model
    model = model_from_json(open('{}.json'.format(model_prefix), 'r').read())
    model.load_weights('{}-best-loss.h5'.format(model_prefix))

    # get all of the examples
    examples, npositives, nnegatives = CollectExamples(prefix, width, radius, subset)

    # get the correspond edges
    edges = CollectEdges(prefix, width, radius, subset)
    assert (len(edges) == examples.shape[0])
    
    # get all of the probabilities 
    probabilities = model.predict_generator(EdgeGenerator(examples, width), examples.shape[0], max_q_size=1000)

    # create the correct labels for the ground truth
    ground_truth = np.zeros(npositives + nnegatives, dtype=np.bool)
    for iv in range(npositives):
        ground_truth[iv] = True

    # get the results with labeled data
    predictions = Prob2Pred(np.squeeze(probabilities[:npositives+nnegatives]))

    # print the confusion matrix
    output_filename = '{}-{}-inference.txt'.format(model_prefix, prefix)
    PrecisionAndRecall(ground_truth, predictions, output_filename)

    # save the probabilities for each edge
    output_filename = '{}-{}.probabilities'.format(model_prefix, prefix)
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('q', examples.shape[0]))
        for ie, (label_one, label_two) in enumerate(edges):
            fd.write(struct.pack('qqd', label_one, label_two, probabilities[ie]))
