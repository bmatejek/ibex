import numpy as np
import time
import struct
import random
import os
from numba import jit

from keras.models import model_from_json

from ibex.cnns.skeleton.util import FindCandidates, ExtractFeature
from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.evaluation.classification import *



# generate candidate features for the predict function
def SkeletonCandidateGenerator(prefix, network_distance, candidates, width, augment):
    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    world_res = dataIO.Resolution(prefix)

    # get the radii for the bounding box in grid coordinates
    radii = (network_distance / world_res[0], network_distance / world_res[1], network_distance / world_res[2])
    index = 0
    
    start_time = time.time()
    continue_printing = True

    # continue indefinitely
    while True:
        # this prevents overflow on the queue - the repeated samples are never used
        if index >= len(candidates): 
            continue_printing = False
            index = 0

        # get the current candidate
        candidate = candidates[index]

        # increment the index
        index += 1

        if continue_printing and not (index % (len(candidates) / 10)): 
            print '{}/{}: {}'.format(index, len(candidates), time.time() - start_time)

        # rotation equals 0
        yield ExtractFeature(segmentation, candidate, width, radii, augment=augment)




# run the forward pass for the given prefix
def Forward(prefix, model_prefix, threshold, maximum_distance, endpoint_distance, network_distance, width, naugmentations):
    # read in the trained model
    model = model_from_json(open('{}.json'.format(model_prefix), 'r').read())
    model.load_weights('{}-best-loss.h5'.format(model_prefix))

    # get the candidate locations 
    positive_candidates = FindCandidates(prefix, threshold, maximum_distance, endpoint_distance, network_distance, 'positive')
    negative_candidates = FindCandidates(prefix, threshold, maximum_distance, endpoint_distance, network_distance, 'negative')
    undetermined_candidates = FindCandidates(prefix, threshold, maximum_distance, endpoint_distance, network_distance, 'undetermined')
    candidates = positive_candidates + negative_candidates + undetermined_candidates
    ncandidates = len(candidates)
    
    # compute augmentations
    probabilities = np.zeros((ncandidates, 1), dtype=np.float64)
    for _ in range(naugmentations):
        probabilities += model.predict_generator(SkeletonCandidateGenerator(prefix, network_distance, candidates, width, True), ncandidates, max_q_size=2000)
    probabilities /= (naugmentations)
    predictions = Prob2Pred(np.squeeze(probabilities))

    
    output_filename = '{}-{}-{}-{}nm-{}nm-{}nm.probabilities'.format(model_prefix, prefix, threshold, maximum_distance, endpoint_distance, network_distance)
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('i', ncandidates))
        for probability in probabilities:
            fd.write(struct.pack('d', probability))

    # create an array of labels
    candidates = positive_candidates + negative_candidates
    ncandidates = len(candidates)
    labels = np.zeros(ncandidates, dtype=np.uint8)
    for ie, candidate in enumerate(candidates):
        labels[ie] = candidate.ground_truth

    # write the precision and recall values
    output_filename = '{}-{}-{}-{}nm-{}nm-{}nm.results'.format(model_prefix, prefix, threshold, maximum_distance, endpoint_distance, network_distance)
    PrecisionAndRecall(labels, predictions[:ncandidates], output_filename)

        
def AnalyzeAugmentation(prefix, model_prefix, threshold, maximum_distance, endpoint_distance, network_distance, width, max_augmentations):
    # read in the trained model
    model = model_from_json(open('{}.json'.format(model_prefix), 'r').read())
    model.load_weights('{}-best-loss.h5'.format(model_prefix))
    
    # get the candidate locations 
    positive_candidates = FindCandidates(prefix, threshold, maximum_distance, endpoint_distance, network_distance, 'positive')
    negative_candidates = FindCandidates(prefix, threshold, maximum_distance, endpoint_distance, network_distance, 'negative')
    candidates = positive_candidates + negative_candidates
    ncandidates = len(candidates)

    # create an array of ground truth labels
    labels = np.zeros(ncandidates, dtype=np.uint8)
    for ie, candidate in enumerate(candidates):
        labels[ie] = candidate.ground_truth
    
    # create the output directory if it does not exist
    model_name = model_prefix.split('/')[2]
    directory = 'plots/test-augmentations/{}'.format(model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # compute augmentations
    probabilities = np.zeros((ncandidates, 1), dtype=np.float64)
    for iv in range(max_augmentations):
        print 'Iteration No. {}'.format(iv)

        # update the probabilities
        probabilities += model.predict_generator(SkeletonCandidateGenerator(prefix, network_distance, candidates, width, True), ncandidates, max_q_size=800)
        
        # get the predictions from the previous probabilities
        predictions = Prob2Pred(np.squeeze(probabilities) / (iv + 1))
        
        # write the precision and recall
        output_filename = '{}/{}-{:04d}.aug'.format(directory, prefix, iv)
        PrecisionAndRecall(labels, predictions, output_filename)
