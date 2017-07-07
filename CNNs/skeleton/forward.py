import time
import struct
import numpy as np
from keras.models import Model, Sequential, model_from_json
from ibex.utilities import dataIO
from ibex.evaluation import classification
from ibex.transforms import seg2seg
from util import FindCandidates, ExtractFeature



# generate candidate features for the predict function
def CandidateGenerator(prefix, segmentation, image, candidates, maximum_distance, window_width, nchannels):
    assert (nchannels == 1 or nchannels == 3 or nchannels == 4)
    # get the grid size and the world resolution in (z, y, x)
    world_res = dataIO.ReadMetaData(prefix)

    # get the radii for the bounding box in grid coordinates
    radii = (maximum_distance / world_res[0], maximum_distance / world_res[1], maximum_distance / world_res[2])

    # keep an index for the number of calls to the generator
    index = 0

    # create a counter for the generator
    start_time = time.time()

    # continue indefinitely
    while index >= 0:
        if (not (index + 1) % 1000):
            print 'Ran {0} iterations in {1:4f} seconds'.format(index + 1, time.time() - start_time)

        if index >= len(candidates):
            break

        # get the current candidate
        candidate = candidates[index]

        # increment the index
        index += 1
        
        # get the labels and the locations for the current candidate
        labels = candidate.Labels()
        location = candidate.Location()

        example = ExtractFeature(segmentation, image, labels, location, radii, window_width, nchannels=nchannels)
        yield example



# create the internal graph structure for multi-cut
def GenerateMultiCutInput(model_prefix, prefix, segmentation, maximum_distance, candidates, probabilities):
    # get the mapping to a smaller set of vertices
    forward_mapping, reverse_mapping = seg2seg.ReduceLabels(segmentation)

    # create multi-cut file
    multicut_filename = 'multicut/{}-{}.graph'.format(model_prefix, prefix)
    
    # open a file to write multi-cut information
    with open(multicut_filename, 'wb') as fd:
        # write the number of vertices and the number of edges
        fd.write(struct.pack('QQ', reverse_mapping.size, len(candidates)))

        # for every merge candidate, determine the weight of the edge
        for ie in range(len(candidates)):
            candidate = candidates[ie]

            # get the probability of merge from neural network
            probability = probabilities[ie]

            # get the labels for these two candidates
            label_one = candidate.label_one
            label_two = candidate.label_two

            # get the new label
            reduced_label_one = forward_mapping[label_one]
            reduced_label_two = forward_mapping[label_two]

            # write the label for both segments and the probability of merge from neural network
            fd.write(struct.pack('QQQQd', label_one, label_two, reduced_label_one, reduced_label_two, probability))



# run the forward pass for the given prefix
def Forward(prefix, maximum_distance, model_prefix, window_width=106, nchannels=1):
    assert (nchannels == 1 or nchannels == 3 or nchannels == 4)

    # read in the trained model
    model = model_from_json(open(model_prefix + '.json', 'r').read())
    model.load_weights(model_prefix + '.h5')

    # get the candidate locations 
    # there is no reason to use the padded data - that is only for training
    candidates = FindCandidates(prefix, maximum_distance, forward=True)
    ncandidates = len(candidates)


    # create an array of labels
    labels = np.zeros(ncandidates, dtype=np.uint8)
    for ie, candidate in enumerate(candidates):
        labels[ie] = candidate.GroundTruth()

    # read in the segmentation file
    segmentation = dataIO.ReadSegmentationData(prefix)

    # get the probabilities, max_q_size = 1 keeps from overflow
    probabilities = model.predict_generator(CandidateGenerator(prefix, segmentation, candidates, maximum_distance, window_width, nchannels), ncandidates, max_q_size=0)
    predictions = classification.prob2pred(probabilities)

    # output the accuracy of this network
    output_filename = '{}-forward.results'.format(model_prefix)

    classification.PrecisionAndRecall(labels, predictions, output_filename)

    output_filename = '{}-thresholds.png'.format(model_prefix)
    classification.ThresholdPredictions(labels, probabilities, output_filename)

    model_prefix = model_prefix.split('/')[1]
    
    GenerateMultiCutInput(model_prefix, prefix, segmentation, maximum_distance, candidates, probabilities)
