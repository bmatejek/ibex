import os
import time
import numpy as np
from ibex.utilities import dataIO
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



# train a convolutional neural network for merging skeletons
def Train(prefix, maximum_distance, output_prefix, window_width=106, nchannels=1, nrotations=8, padding=0, batch_size=2):
    # make sure the number of channels is 1 or 3
    assert (nchannels == 1 or nchannels == 3)

    # make sure the batch size is even 
    assert (batch_size % 2 == 0)

    # make sure a folder for the output prefix exists
    root_location = output_prefix.rfind('/')
    output_folder = output_prefix[:root_location]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # read in the h5 segmentation file
    segmentation = dataIO.ReadSegmentationData(prefix)

    # get the grid size and the world resolution in (z, y, x)
    grid_size = segmentation.shape
    world_res = dataIO.ReadMetaData(prefix)

    # get the radii for the bounding box in grid coordinates
    radii = (maximum_distance / world_res[0], maximum_distance / world_res[1], maximum_distance / world_res[2])

    # get all of the candidates for this prefix
    candidates = FindCandidates(prefix, maximum_distance, padding, forward=False)
    ncandidates = len(candidates)
    
    # create the model
    model = Sequential()

    AddConvolutionLayer(model, 16, (3, 3, 3), padding='valid', activation='relu', input_shape=(window_width, window_width, window_width, nchannels))
    AddConvolutionLayer(model, 16, (3, 3, 3), padding='valid', activation='relu')
    AddPoolingLayer(model, (2, 2, 2), dropout=0.00)

    if window_width > 100:
        AddConvolutionLayer(model, 32, (3, 3, 3), padding='valid', activation='relu')
        AddConvolutionLayer(model, 32, (3, 3, 3), padding='valid', activation='relu')
        AddPoolingLayer(model, (2, 2, 2), dropout=0.00)

    AddConvolutionLayer(model, 64, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 64, (3, 3, 3), padding='valid', activation='relu')
    AddPoolingLayer(model, (2, 2, 2), dropout=0.00)
    
    AddConvolutionLayer(model, 128, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 128, (3, 3, 3), padding='valid', activation='relu')
    AddPoolingLayer(model, (2, 2, 2), dropout=0.00)

    AddFlattenLayer(model)
    AddDenseLayer(model, 512, dropout=0.00, activation='relu')
    AddDenseLayer(model, 1, dropout=0.00, activation='sigmoid')

    # create an initial learning rate with a decay factor
    initial_learning_rate=1e-4
    decay_rate=5e-8

    # compile the model
    adm = Adam(lr=initial_learning_rate, beta_1=0.99, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=adm)

    # if the number of epochs is -1, run once for every example and every permutation
    if num_epochs == -1:
        if not (nrotations * ncandidates) % batch_size:
            num_epochs = nrotations * ncandidates / batch_size
        else:
            num_epochs = nrotations * ncandidates / batch_size + 1

    # keep track of the number of epochs here separately
    # this may reset to 0 if number of examples reached
    index = 0

    # run for all epochs and time for every group of 20
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        if not epoch % 20:
            print '{0}/{1} in {2:4f} seconds'.format(epoch, num_epochs, time.time() - start_time)
            start_time = time.time()

        # create arrays for the examples and labels
        examples = np.zeros((batch_size, window_width, window_width, window_width, nchannels))
        labels = np.zeros((batch_size, 1))

        # iterate over the entire batch
        for ib in range(batch_size):
            # get the index and the rotation
            candidate_rotation = index / ncandidates
            candidate_index = index % ncandidates

            # retrieve the actual candidate
            candidate = candidates[candidate_index]

            # get the information about this candidate
            labels = candidate.Labels()
            location = candidate.Location()
            
            # get the example for this candidate
            example = ExtractFeature(segmentation, labels, location, radii, window_width, candidate_rotation, nchannels, padding)

            examples[ib,:,:,:,:] = example
            labels[ib,:] = candidate.GroundTruth()
            
            index += 1

            # provide overflow relief
            if index >= ncandidates * nrotations:
                index = 0

        # TODO delete - guarantee equal proportions
        positive_examples = 0
        negative_examples = 0
        for ib in range(batch_size):
            if labels[ib,0]:
                positive_examples += 1
            else:
                negative_examples += 1
        assert (positive_examples == negative_examples)


        # update the learning rate
        current_learning_rate = initial_learning_rate / (1.0 + (epoch - 1) * decay_rate)
        backend.set_value(model.optimizer.lr, current_learning_rate)

        # fit the model
        model.fit(examples, labels, epochs=1, verbose=0)

        # save for every 1000 examples seen
        if not epoch % (1000 / batch_size):
            # save an indermediate model
            json_string = model.to_json()
            open(output_prefix + '-' + str(epoch) + '.json', 'w').write(json_string)
            model.save_weights(output_prefix + '-' + str(epoch) + '.h5')

    # save the fully trained model
    json_string = model.to_json()
    open(output_prefix + '.json', 'w').write(json_string)
    model.save_weights(output_prefix + '.h5')
