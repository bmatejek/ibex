import time
import struct
import numpy as np
from keras.models import Model, Sequential, model_from_json
from util import FindCandidates, ExtractFeature
from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.evaluation import classification



# generate candidate features for the prediction function
def CandidateGenerator(prefix_one, prefix_two, maximum_distance, candidates, width, nchannels):
    # read in the segmentation file
    segmentation_one = dataIO.ReadSegmentationData(prefix_one)
    segmentation_two = dataIO.ReadSegmentationData(prefix_two)
    image_one = dataIO.ReadImageData(prefix_one)
    image_two = dataIO.ReadImageData(prefix_two)
    bbox_one = dataIO.GetWorldBBox(prefix_one)
    bbox_two = dataIO.GetWorldBBox(prefix_two)
    world_res = dataIO.Resolution(prefix_one)
    assert (world_res == dataIO.Resolution(prefix_two))

    # get the radii for the bounding box in grid coordinates
    radii = (maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X])   

    # keep an index for the number of calls to generator
    index = 0

    # start statistics
    start_time = time.time()
    while True:
        if not (index + 1) % 1000:
            print 'Ran {} iterations in {:4f} seconds'.format(index + 1, time.time() - start_time)
        # prevent overflow on queue
        if index >= len(candidates):
            index = 0

        # get the current candidate
        candidate = candidates[index]
        # increment the index
        index += 1

        # get this feature - rotation = 0 
        example = ExtractFeature(segmentation_one, segmentation_two, image_one, image_two, bbox_one, bbox_two, candidate, radii, width, 0, nchannels)
        yield example



# run the forward pass for the given prefixes
def Forward(prefix_one, prefix_two, threshold, maximum_distance, model_prefix, width):
    # constants
    nchannels = 3

    # read in the trained model
    model = model_from_json(open('{}.json'.format(model_prefix), 'r').read())
    model.load_weights('{}.h5'.format(model_prefix))

    # get the candidate locations
    candidates = FindCandidates(prefix_one, prefix_two, threshold, maximum_distance)
    ncandidates = len(candidates)

    # create an array of labels
    labels = np.zeros(ncandidates, dtype=np.uint8)
    for ie, candidate in enumerate(candidates):
        labels[ie] = candidate.GroundTruth()


    # get the probabilities 
    probabilities = model.predict_generator(CandidateGenerator(prefix_one, prefix_two, maximum_distance, candidates, width, nchannels), ncandidates, max_q_size=20)
    predictions = classification.prob2pred(probabilities)    

    # output the accuracy of this network
    output_filename = '{}-forward.results'.format(model_prefix)

    classification.PrecisionAndRecall(labels, predictions, output_filename)

    output_filename = '{}-threshold.png'.format(model_prefix)
    classification.ThresholdPredictions(labels, probabilities, output_filename)