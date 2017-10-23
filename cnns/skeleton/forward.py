import numpy as np
import time
import struct
import random
import os
import natsort

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
        if not ((index + 1) % 1000): 
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
def LearningCurve(prefix, model_prefix, threshold, maximum_distance, network_distance, width, parameters, nsamples=400):
    # get the candidate locations 
    candidates = FindCandidates(prefix, threshold, maximum_distance, network_distance, inference=True)

    errors = []
    epochs = []

    # how often to get the batch size
    test_frequency = 5
    
    # make sure the folder for the model prefix exists
    root_location = model_prefix.rfind('/')
    output_folder = model_prefix[:root_location]
    
    # get the list of saved models
    saved_models = [model[:-3] for model in natsort.natsorted(os.listdir(output_folder)) if model.endswith('.h5')]

    # go through every epoch for a given frequency
    for ie, saved_model in enumerate(saved_models):
        if (ie == len(saved_models) - 1) or (ie % test_frequency): continue
        start_time = time.time()

        epoch = int(saved_model.split('-')[1])

        model = model_from_json(open('{}/{}.json'.format(output_folder, saved_model), 'r').read())
        model.load_weights('{}/{}.h5'.format(output_folder, saved_model))

        random.shuffle(candidates)

        # get the probabilities
        probabilities = model.predict_generator(SkeletonCandidateGenerator(prefix, network_distance, candidates, width), nsamples, max_q_size=200)
        predictions = Prob2Pred(np.squeeze(probabilities))

        # create an array of labels
        ncorrect = 0
        labels = np.zeros(nsamples, dtype=np.uint8)
        for ie in range(nsamples):
            labels[ie] = candidates[ie].ground_truth
            if predictions[ie] == labels[ie]: ncorrect += 1

        # print results
        errors.append(1.0 - ncorrect / float(nsamples))
        epochs.append(epoch)

        print '[{} Loss = {} Total Time = {}]'.format(saved_model, 1.0 - ncorrect / float(nsamples), time.time() - start_time)

    # plot the curve
    plt.plot(epochs, errors, 'g')
    plt.axis([0, max(epochs), 0, 1.0])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training Curve {}'.format(prefix))

    plt.savefig('{}-{}-training-curve.png'.format(model_prefix, prefix))

    plt.clf()
