import numpy as np
import time
import struct

from keras.models import Model, Sequential, model_from_json

from ibex.cnns.skeleton.util import FindCandidates, ExtractFeature
from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.evaluation.classification import *
import matplotlib.pyplot as plt



# generate candidate features for the predict function
def SkeletonCandidateGenerator(prefix, network_distance, candidates, width):
    start_time = time.time()
    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    world_res = dataIO.Resolution(prefix)

    # get the radii for the bounding box in grid coordinates
    radii = (network_distance / world_res[0], network_distance / world_res[1], network_distance / world_res[2])
    index = 0

    # continue indefinitely
    while True:
        if not ((index + 1) % 100): 
            print '{}/{}: {}'.format(index + 1,  len(candidates), time.time() - start_time)
        # this prevents overflow on the queue - the repeated samples are never used
        if index >= len(candidates): index = 0

        # get the current candidate
        candidate = candidates[index]

        # increment the index
        index += 1

        # rotation equals 0
        yield ExtractFeature(segmentation, candidate, width, radii, 0)




# run the forward pass for the given prefix
def Forward(prefix, model_prefix, threshold, maximum_distance, network_distance, width, parameters):
    # read in the trained model
    model = model_from_json(open('{}.json'.format(model_prefix), 'r').read())
    model.load_weights('{}.h5'.format(model_prefix))
    
    # get the candidate locations 
    candidates = FindCandidates(prefix, threshold, maximum_distance, network_distance, inference=True)
    ncandidates = len(candidates)

    # get the probabilities
    probabilities = model.predict_generator(SkeletonCandidateGenerator(prefix, network_distance, candidates, width), ncandidates, max_q_size=200)
    predictions = Prob2Pred(np.squeeze(probabilities))

    # create an array of labels
    labels = np.zeros(ncandidates, dtype=np.uint8)
    for ie, candidate in enumerate(candidates):
        if (ie >= ncandidates): continue
        labels[ie] = candidate.ground_truth

    # write the precision and recall values
    output_filename = '{}-{}-{}-{}nm.results'.format(model_prefix, prefix, threshold, maximum_distance)
    PrecisionAndRecall(labels, predictions, output_filename)

    output_filename = '{}-{}-{}-{}nm.probabilities'.format(model_prefix, prefix, threshold, maximum_distance)
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('i', ncandidates))
        for probability in probabilities:
            fd.write(struct.pack('d', probability))

    # output the probabilities for the network
    output_filename = 'results/skeleton/{}-{}-{}nm.results'.format(prefix, threshold, maximum_distance)
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('i', ncandidates))
        for probability in probabilities:
            fd.write(struct.pack('d', probability))



# generate a training curve
def LearningCurve(prefix, model_prefix, threshold, maximum_distance, network_distance, width, parameters, nsamples=200):
    # get the candidate locations 
    candidates = FindCandidates(prefix, threshold, maximum_distance, network_distance, inference=True)
    ncandidates = len(candidates)

    # get relevant parameters
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']

    # find out how many epochs there are
    if parameters['augment']: rotations = 16
    else: rotations = 1
    if rotations * ncandidates % batch_size:
        max_epoch = (iterations * rotations * ncandidates / batch_size) + 1
    else:
        max_epoch = (iterations * rotations * ncandidates / batch_size)

    errors = []
    epochs = []

    # how often to get the batch size
    test_frequency = 1000 / batch_size
    
    # go through every epoch for a given frequency
    for epoch in range(test_frequency, max_epoch, test_frequency):
        # read in the trained model
        model = model_from_json(open('{}-{}.json'.format(model_prefix, epoch), 'r').read())
        model.load_weights('{}-{}.h5'.format(model_prefix, epoch))

        random.shuffle(candidates)

        # get the probabilities
        probabilities = model.predict_generator(SkeletonCandidateGenerator(prefix, network_distance, candidates, width), nsamples, max_q_size=200)
        predictions = Prob2Pred(probabilities)

        # create an array of labels
        ncorrect = 0
        labels = np.zeros(nsamples, dtype=np.uint8)
        for ie in range(nsamples):
            labels[ie] = candidates[ie].ground_truth
            if predictions[ie] == labels[ie]: ncorrect += 1

        # print results
        errors.append(1.0 - ncorrect / float(nsamples))
        epochs.append(epoch)

    # plot the curve
    plt.plot(epochs, errors, 'g')
    plt.axis([0, max_epoch + test_frequency, 0, 1.0])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training Curve {}'.format(prefix))

    plt.savefig('{}-{}-training-curve.png'.format(model_prefix, prefix))
