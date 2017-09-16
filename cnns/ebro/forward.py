import struct
import time
import numpy as np

from keras.models import model_from_json

from ibex.cnns.ebro.util import ExtractFeature, FindCandidates
from ibex.evaluation.classification import *
from ibex.utilities.constants import *
from ibex.utilities import dataIO



# generate candidates for inference
def EbroCandidateGenerator(prefix_one, prefix_two, maximum_distance, candidates, width):
    # read in all relevant information
    segmentations = (dataIO.ReadSegmentationData(prefix_one), dataIO.ReadSegmentationData(prefix_two))
    assert (segmentations[0].shape == segmentations[1].shape)
    images = (dataIO.ReadImageData(prefix_one), dataIO.ReadImageData(prefix_two))
    assert (images[0].shape == images[1].shape)
    bboxes = (dataIO.GetWorldBBox(prefix_one), dataIO.GetWorldBBox(prefix_two))
    world_res = dataIO.Resolution(prefix_one)
    assert (world_res == dataIO.Resolution(prefix_two))

    # get the radii for the relevant region
    radii = (maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X])
    index = 0
    start_time = time.time()
    while True:
        if not ((index + 1) % 100): 
            print '{}/{}: {}'.format(index + 1,  len(candidates), time.time() - start_time)

        # prevent overflow
        if index >= len(candidates): index = 0

        candidate = candidates[index]
        index += 1

        # rotation equals 0
        yield ExtractFeature(segmentations, images, bboxes, candidate, width, radii, 0)



# run inference on the neural network for these prefixes
def Forward(prefix_one, prefix_two, model_prefix, threshold, maximum_distance, width):
    # read in the trained model
    model = model_from_json(open('{}.json'.format(model_prefix), 'r').read())
    model.load_weights('{}.h5'.format(model_prefix))

    # get the candidate locations
    candidates = FindCandidates(prefix_one, prefix_two, threshold, maximum_distance, inference=True)
    ncandidates = len(candidates)
    
    # get the probabilities
    probabilities = model.predict_generator(EbroCandidateGenerator(prefix_one, prefix_two, maximum_distance, candidates, width), ncandidates, max_queue_size=20)
    predictions = Prob2Pred(probabilities)

    # create an array of labels
    labels = np.zeros(ncandidates, dtype=np.uint8)
    for iv, candidate in enumerate(candidates):
        labels[iv] = candidate.ground_truth

    # output the accuracy of this network
    output_filename = '{}-{}-{}-{}-{}nm.results'.format(model_prefix, prefix_one, prefix_two, threshold, maximum_distance)
    PrecisionAndRecall(labels, predictions, output_filename)

    output_filename = '{}-{}-{}-{}-{}nm.probabilities'.format(model_prefix, prefix_one, prefix_two, threshold, maximum_distance)
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('i', ncandidates))
        for probability in probabilities:
            fd.write(struct.pack('d', probability))

    # output the probabilities for the network
    output_filename = 'results/ebro/{}-{}-{}-{}nm.results'.format(prefix_one, prefix_two, threshold, maximum_distance)
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('i', ncandidates))
        for probability in probabilities:
            fd.write(struct.pack('d', probability))
