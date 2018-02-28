import numpy as np
import time
import struct
import random
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
    # continue indefinitely
    while True:
        # this prevents overflow on the queue - the repeated samples are never used
        if index >= len(candidates): index = 0

        # get the current candidate
        candidate = candidates[index]

        # increment the index
        index += 1

        if not (index % 1000): 
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
    candidates = positive_candidates + negative_candidates
    ncandidates = len(candidates)

    # compute augmentations
    probabilities = model.predict_generator(SkeletonCandidateGenerator(prefix, network_distance, candidates, width, False), ncandidates, max_q_size=200)
    for _ in range(naugmentations):
        probabilities += model.predict_generator(SkeletonCandidateGenerator(prefix, network_distance, candidates, width, True), ncandidates, max_q_size=200)
    probabilities /= (1 + naugmentations)
    predictions = Prob2Pred(np.squeeze(probabilities))

    # create an array of labels
    labels = np.zeros(ncandidates, dtype=np.uint8)
    for ie, candidate in enumerate(candidates):
        labels[ie] = candidate.ground_truth

    # write the precision and recall values
    output_filename = '{}-{}-{}-{}nm.results'.format(model_prefix, prefix, threshold, maximum_distance)
    PrecisionAndRecall(labels, predictions, output_filename)

    output_filename = '{}-{}-{}-{}nm.probabilities'.format(model_prefix, prefix, threshold, maximum_distance)
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('i', ncandidates))
        for probability in probabilities:
            fd.write(struct.pack('d', probability))


@jit(nopython=True)
def ComparePredictions(previous_predictions, current_predictions, labels):
    ncandidates = current_predictions.size
    ncorrections = 0
    nerrors = 0
    
    for iv in range(ncandidates):
        if current_predictions[iv] == previous_predictions[iv]: continue

        if current_predictions[iv] == labels[iv]: ncorrections += 1
        else: nerrors += 1

    return ncorrections, nerrors


        
def AnalyzeAugmentation(prefix, model_prefix, threshold, maximum_distance, endpoint_distance, network_distance, width):
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
    
    # compute augmentations
    probabilities = model.predict_generator(SkeletonCandidateGenerator(prefix, network_distance, candidates, width, False), ncandidates, max_q_size=200)
    first_predictions = Prob2Pred(np.squeeze(probabilities))
    for iv in range(100):
        # get the predictions from the previous probabilities
        previous_predictions = Prob2Pred(np.squeeze(probabilities) / (iv + 1))
        # update the probabilities
        probabilities += model.predict_generator(SkeletonCandidateGenerator(prefix, network_distance, candidates, width, True), ncandidates, max_q_size=200)
        # get the predictions after this round
        current_predictions = Prob2Pred(np.squeeze(probabilities) / (iv + 2))
        
        # find which predictions are different
        ncorrections, nerrors = ComparePredictions(previous_predictions, current_predictions, labels)

        print 'Augmentation No. {}'.format(iv)
        print '  From Previous Iteration'
        print '    Corrections: {}'.format(ncorrections)
        print '    Errors: {}'.format(nerrors)

        ncorrections, nerrors = ComparePredictions(first_predictions, current_predictions, labels)

        print 'Augmentation No. {}'.format(iv)
        print '  From First Iteration'
        print '    Corrections: {}'.format(ncorrections)
        print '    Errors: {}'.format(nerrors)

        
