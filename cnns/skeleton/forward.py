import numpy as np
import time
import struct
import random
import os
import natsort

from keras.models import model_from_json

from ibex.cnns.skeleton.util import FindCandidates, ExtractFeature
from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.evaluation.classification import *



# generate candidate features for the predict function
def SkeletonCandidateGenerator(prefix, network_distance, candidates, width):
    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    world_res = dataIO.Resolution(prefix)

    # get the radii for the bounding box in grid coordinates
    radii = (network_distance / world_res[0], network_distance / world_res[1], network_distance / world_res[2])
    index = 0
    
    start_time = time.time()
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
    model.load_weights('{}-best-loss.h5'.format(model_prefix))
    
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
