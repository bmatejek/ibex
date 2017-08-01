import time
import struct
import numpy as np
from keras.models import Model, Sequential, model_from_json
from ibex.utilities import dataIO
from ibex.evaluation.classification import *
from ibex.transforms import seg2seg
from util import FindCandidates, ExtractFeature



# generate candidate features for the predict function
def SkeletonCandidateGenerator(prefix, maximum_distance, candidates, width):
    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    world_res = dataIO.Resolution(prefix)

    # get the radii for the bounding box in grid coordinates
    radii = (maximum_distance / world_res[0], maximum_distance / world_res[1], maximum_distance / world_res[2])
    index = 0

    # continue indefinitely
    while True:
        # this prevents overflow on the queue - the repeated samples are never used
        if index >= len(candidates): index = 0

        # get the current candidate
        candidate = candidates[index]

        # increment the index
        index += 1

        # rotation equals 0
        yield ExtractFeature(segmentation, candidate, width, radii, 0)



# # create the internal graph structure for multi-cut
# def GenerateMultiCutInput(model_prefix, prefix, segmentation, maximum_distance, candidates, probabilities):
#     # get the mapping to a smaller set of vertices
#     forward_mapping, reverse_mapping = seg2seg.ReduceLabels(segmentation)

#     # create multi-cut file
#     multicut_filename = 'multicut/{}-{}.graph'.format(model_prefix, prefix)
    
#     # open a file to write multi-cut information
#     with open(multicut_filename, 'wb') as fd:
#         # write the number of vertices and the number of edges
#         fd.write(struct.pack('QQ', reverse_mapping.size, len(candidates)))

#         # for every merge candidate, determine the weight of the edge
#         for ie in range(len(candidates)):
#             candidate = candidates[ie]

#             # get the probability of merge from neural network
#             probability = probabilities[ie]

#             # get the labels for these two candidates
#             label_one = candidate.label_one
#             label_two = candidate.label_two

#             # get the new label
#             reduced_label_one = forward_mapping[label_one]
#             reduced_label_two = forward_mapping[label_two]

#             # write the label for both segments and the probability of merge from neural network
#             fd.write(struct.pack('QQQQd', label_one, label_two, reduced_label_one, reduced_label_two, probability))



# run the forward pass for the given prefix
def Forward(prefix, model_prefix, maximum_distance, width):
    # read in the trained model
    model = model_from_json(open(model_prefix + '.json', 'r').read())
    model.load_weights(model_prefix + '.h5')

    # get the candidate locations 
    candidates = FindCandidates(prefix, maximum_distance, inference=True)
    ncandidates = len(candidates)

    # get the probabilities
    probabilities = model.predict_generator(SkeletonCandidateGenerator(prefix, maximum_distance, candidates, width), ncandidates, max_q_size=20)
    predictions = Prob2Pred(probabilities)

    # create an array of labels
    labels = np.zeros(ncandidates, dtype=np.uint8)
    for ie, candidate in enumerate(candidates):
        labels[ie] = candidate.ground_truth


    # output the accuracy of this network
    output_filename = '{}-{}-forward.results'.format(model_prefix, prefix)
    PrecisionAndRecall(labels, predictions, output_filename)
