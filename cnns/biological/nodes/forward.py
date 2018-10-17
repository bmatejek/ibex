import os

from keras.models import model_from_json

from ibex.cnns.biological.nodes.util import AugmentFeature


def NodeGenerator(filenames, width):
    filename_index = 0

    while True:
        # prevent overflow of the queue
        if filename_index == len(filenames):
            filename_index = 0

        candidate = dataIO.ReadH5File(filenames[filename_index], 'main')
        yield AugmentFeature(candidate, width)



def Forward(prefix, model_prefix, width, radius, subset):
    # read in the trained model
    model = model_from_json(open('{}.json'.format(model_prefix), 'r').read())
    model.load_weights('{}-best-loss.h5'.format(model_prefix))

    # get all of the candidates for this prefix
    positive_directory = 'features/biological/nodes-{}nm/{}/positives'.format(radius, subset)
    negative_directory = 'features/biological/nodes-{}nm/{}/negatives'.format(radius, subset)
    unknowns_directory = 'features/biological/nodes-{}nm/{}/unknowns'.format(radius, subset)

    # get the positive, negative, and unknown locations for this directory
    positive_filenames = []
    for filename in os.listdir(positive_directory):
        if prefix in filename:
            positive_filenames.append(filename)

    negative_filenames = []
    for filename in os.listdir(negative_directory):
        if prefix in filename:
            negative_filenames.append(filename)

    unknowns_filenames = []
    for filename in os.listdir(unknowns_directory):
        if prefix in filename:
            unknowns_filenames.append(filename)

    filenames = positive_filenames + negative_filenames + unknowns_filenames

    probabilities = model.predict_generator(NodeGenerator(filenames, width), len(filenames), max_q_size=2000)
    
    # get some results on the known quantities
    nknowns = len(positive_filenames) + len(negative_filenames)
    predictions = Prob2Pred(np.squeeze(probabilities[:knownns]))
    labels = np.zeros(nknowns, dtype=np.bool)
    for iv in range(len(positive_filenames)):
        labels[iv] = True

    PrecisionAndRecall(labels, predictions)