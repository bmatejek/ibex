import os
import time
import numpy as np
from ibex.utilities import dataIO
from ibex.utilities.constants import *
from util import FindCandidates, ExtractFeature
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution3D, MaxPooling3D
from keras import backend



# add a convolutional layer to the model
def AddConvolutionLayer(model, filter_size, kernel_size=(3,3,3), padding='valid', activation='relu', input_shape=None):
    # get the current level
    level = len(model.layers)
    # add this convolution layer
    if not input_shape == None:
        model.add(Convolution3D(filter_size, kernel_size, padding=padding, input_shape=input_shape))
    else:
        model.add(Convolution3D(filter_size, kernel_size, padding=padding))
    # print out input and output shape size
    print 'Convolution Layer: ' + str(model.layers[level].input_shape) + ' -> ' + str(model.layers[level].output_shape)
    # add an activation layer
    model.add(Activation(activation))



# add a 3d pooling layer to the model
def AddPoolingLayer(model, pool_size=(2,2,2), dropout=0.25):
    # get the current level
    level = len(model.layers)
    # add the max pooling layer
    model.add(MaxPooling3D(pool_size=pool_size))
    # print out input and output shape size
    print 'Max Pooling Layer: ' + str(model.layers[level].input_shape) + ' -> ' + str(model.layers[level].output_shape)
    # add a dropout layer
    if (dropout > 0.0):
        model.add(Dropout(dropout))



# add a flattening layer to the model
def AddFlattenLayer(model):
    # get the current level
    level = len(model.layers)
    # flatten the model
    model.add(Flatten())
    # print out input and output shape
    print 'Flatten Layer: ' + str(model.layers[level].input_shape) + ' -> ' + str(model.layers[level].output_shape)



# add a dense layer to the model
def AddDenseLayer(model, filter_size, dropout, activation):
    # get the current level
    level = len(model.layers)
    # add a dense layer
    model.add(Dense(filter_size))
    # add a dropout layer
    if (dropout > 0.0):
        model.add(Dropout(dropout))
    # add an activation layer
    model.add(Activation(activation))
    # print out input and output shape
    print 'Dense Layer: ' + str(model.layers[level].input_shape) + ' -> ' + str(model.layers[level].output_shape)



# train a convolutional neural network for boundary locations
def Train(prefix_one, prefix_two, threshold, maximum_distance, output_prefix, width):
    # constants for training
    starting_epoch = 1
    nchannels = 4
    batch_size = 2
    nrotations = 16
    niterations = 2

    # make sure a folder for the output prefix exists
    root_location = output_prefix.rfind('/')
    output_folder = output_prefix[:root_location]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # read in both segmentation and image files
    segmentation_one = dataIO.ReadSegmentationData(prefix_one)
    segmentation_two = dataIO.ReadSegmentationData(prefix_two)
    image_one = dataIO.ReadImageData(prefix_one)
    image_two = dataIO.ReadImageData(prefix_two)
    bbox_one = dataIO.GetWorldBBox(prefix_one)
    bbox_two = dataIO.GetWorldBBox(prefix_two)

    # get the grid size and the world resolution 
    grid_size = segmentation_one.shape
    assert (segmentation_one.shape == segmentation_two.shape)
    world_res = dataIO.Resolution(prefix_one)
    assert (world_res == dataIO.Resolution(prefix_two))

    # get the radii for the bounding box in grid coordinates
    radii = (maximum_distance / world_res[IB_Z], maximum_distance / world_res[IB_Y], maximum_distance / world_res[IB_X])

    # get all of the candidates for this prefixes
    candidates = FindCandidates(prefix_one, prefix_two, threshold, maximum_distance)
    ncandidates = len(candidates)

    # create the model
    model = Sequential()

    AddConvolutionLayer(model, 16, (3, 3, 3), padding='valid', activation='relu', input_shape=(width[IB_Z], width[IB_Y], width[IB_X], nchannels))
    AddConvolutionLayer(model, 16, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 16, (3, 3, 3), padding='valid', activation='relu')
    AddPoolingLayer(model, (1, 2, 2), dropout=0.00)

    AddConvolutionLayer(model, 32, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 32, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 32, (3, 3, 3), padding='valid', activation='relu')
    AddPoolingLayer(model, (1, 2, 2), dropout=0.00)

    AddConvolutionLayer(model, 64, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 64, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 64, (3, 3, 3), padding='valid', activation='relu')
    AddPoolingLayer(model, (2, 2, 2), dropout=0.00)

    AddConvolutionLayer(model, 128, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 128, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 128, (3, 3, 3), padding='valid', activation='relu')
    AddPoolingLayer(model, (2, 2, 2), dropout=0.00)

    AddFlattenLayer(model)
    AddDenseLayer(model, 512, dropout=0.00, activation='relu')
    AddDenseLayer(model, 1, dropout=0.00, activation='sigmoid')

    # initial learning rate and decay rates
    initial_learning_rate = 1e-4
    decay_rate = 5e-8

    # compile the model
    adm = Adam(lr=initial_learning_rate, beta_1=0.99, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=adm)

    # determine the total number of epochs
    if not nrotations * ncandidates % batch_size:
        num_epochs = (niterations * nrotations * ncandidates / batch_size)
    else:
        num_epochs = (niterations * nrotations * ncandidates / batch_size) + 1

    # keep track of the index (needs to reset when niterations != 1)
    index = 0

    # use a pretrained network?
    if not starting_epoch == 1:
        # update the decay rate
        example_pairs = starting_epoch * batch_size / 2
        current_learning_rate = initial_learning_rate / (1.0 + example_pairs * decay_rate)
        backend.set_value(model.optimizer.lr, current_learning_rate)

        # set the index
        index = (starting_epoch * batch_size) % (ncandidates * nrotations)

        # load the model weights
        model.load_weights('{}-{}.h5'.format(output_prefix, starting_epoch))

    # run all epochs and time for every group of 20
    start_time = time.time()
    for epoch in range(starting_epoch, num_epochs + 1):
        # print statistics
        if not epoch % 20:
            print '{}/{} in {:4f} seconds'.format(epoch, num_epochs, time.time() - start_time)
            start_time = time.time()

        # create arrays for the examples and labels
        examples = np.zeros((batch_size, width[IB_Z], width[IB_Y], width[IB_X], nchannels))
        labels = np.zeros((batch_size, 1))

        # iterate over the entire batch
        for ib in range(batch_size):
            # get the index and the rotation
            candidate_rotation = index / ncandidates
            candidate_index = index % ncandidates

            # retrieve the actual candidate
            candidate = candidates[candidate_index]

            # get the information about this candidate
            candidate_labels = candidate.Labels()
            candidate_location = candidate.Location()

            # get the feature for this candidate
            example = ExtractFeature(segmentation_one, segmentation_two, image_one, image_two, bbox_one, bbox_two, candidate, radii, width, candidate_rotation, nchannels)

            examples[ib,:,:,:,:] = example
            labels[ib,:] = candidate.GroundTruth()

            index += 1

            # provide overflow relief
            if index >= ncandidates * nrotations:
                index = 0

        # fit the model
        model.fit(examples, labels, epochs=1, verbose=0)

        # save for every 1000 examples seen
        if not epoch % (1000 / batch_size):
            # save the intermediate model
            json_string = model.to_json()
            open('{}-{}.json'.format(output_prefix, epoch), 'w').write(json_string)
            model.save_weights('{}-{}.h5'.format(output_prefix, epoch))

        # update the learning rate
        example_pairs = epoch * batch_size / 2
        current_learning_rate = initial_learning_rate / (1.0 + example_pairs * decay_rate)
        backend.set_value(model.optimizer.lr, current_learning_rate)

    # save the fully trained model
    json_string = model.to_json()
    open('{}.json'.format(output_prefix)).write(json_string)
    model.save_weights('{}.h5'.format(output_prefix))