import time
import struct
import numpy as np

from keras.models import Model, Sequential, model_from_json

from ibex.utilities import dataIO
from ibex.evaluation.classification import *
from ibex.transforms import seg2seg
from ibex.cnns.skeleton.util import FindCandidates, ExtractFeature



# generate candidate features for the predict function
def SkeletonCandidateGenerator(prefix, window_radius, candidates, width):
    start_time = time.time()
    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    world_res = dataIO.Resolution(prefix)

    # get the radii for the bounding box in grid coordinates
    radii = (window_radius / world_res[0], window_radius / world_res[1], window_radius / world_res[2])
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
def Forward(prefix, model_prefix, threshold, maximum_distance, window_radius, width):
    # read in the trained model
    model = model_from_json(open(model_prefix + '.json', 'r').read())
    model.load_weights(model_prefix + '.h5')

    # get the candidate locations 
    candidates = FindCandidates(prefix, threshold, maximum_distance, inference=True)
    ncandidates = len(candidates)

    # get the probabilities
    probabilities = model.predict_generator(SkeletonCandidateGenerator(prefix, window_radius, candidates, width), ncandidates, max_q_size=20)
    assert (probabilities.size == ncandidates)
    predictions = Prob2Pred(probabilities)

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

    # output the probabilities for the network
    output_filename = 'results/skeleton/{}-{}-{}nm.results'.format(prefix, threshold, maximum_distance)
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('i', ncandidates))
        for probability in probabilities:
            fd.write(struct.pack('d', probability))
