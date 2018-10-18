import os
import time
import numpy as np

from keras.models import model_from_json

from ibex.utilities import dataIO
from ibex.utilities.constants import *
from ibex.cnns.biological.nodes.util import AugmentFeature
from ibex.evaluation.classification import Prob2Pred, PrecisionAndRecall

def NodeGenerator(filenames, width):
    filename_index = 0
    start_time = time.time()
    while True:
        # prevent overflow of the queue
        if filename_index == len(filenames):
            filename_index = 0

        if (filename_index + 1) % 1000 == 0:
            print '{}/{} in {:0.4f} seconds'.format(filename_index, len(filenames), time.time() - start_time)
                                    
        candidate = dataIO.ReadH5File(filenames[filename_index], 'main')

        # update the filename index
        filename_index += 1
        
        yield AugmentFeature(candidate, width)



def Forward(prefix, model_prefix, width, radius, subset):
    # read in the trained model
    model = model_from_json(open('{}.json'.format(model_prefix), 'r').read())
    model.load_weights('{}-best-loss.h5'.format(model_prefix))

    # get all of the candidates for this prefix
    positive_directory = 'features/biological/nodes-{}nm-{}x{}x{}/{}/positives'.format(radius, width[IB_Z+1], width[IB_Y+1], width[IB_X+1], subset)
    negative_directory = 'features/biological/nodes-{}nm-{}x{}x{}/{}/negatives'.format(radius, width[IB_Z+1], width[IB_Y+1], width[IB_X+1], subset)
    unknowns_directory = 'features/biological/nodes-{}nm-{}x{}x{}/{}/unknowns'.format(radius, width[IB_Z+1], width[IB_Y+1], width[IB_X+1], subset)

    # get the positive, negative, and unknown locations for this directory
    positive_filenames = []
    for filename in os.listdir(positive_directory):
        if prefix in filename:
            positive_filenames.append('{}/{}'.format(positive_directory, filename))

    negative_filenames = []
    for filename in os.listdir(negative_directory):
        if prefix in filename:
            negative_filenames.append('{}/{}'.format(negative_directory, filename))

    unknowns_filenames = []
    for filename in os.listdir(unknowns_directory):
        if prefix in filename:
            unknowns_filenames.append('{}/{}'.format(unknowns_directory, filename))

    filenames = positive_filenames + negative_filenames# + unknowns_filenames

    probabilities = model.predict_generator(NodeGenerator(filenames, width), len(filenames), max_q_size=2000)
    
    # get some results on the known quantities
    nknowns = len(positive_filenames) + len(negative_filenames)
    predictions = Prob2Pred(np.squeeze(probabilities[:nknowns]))
    labels = np.zeros(nknowns, dtype=np.bool)
    for iv in range(len(positive_filenames)):
        labels[iv] = True

    PrecisionAndRecall(labels, predictions)
