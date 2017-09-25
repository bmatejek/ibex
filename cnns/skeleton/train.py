import os
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras import backend

from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.cnns.skeleton.util import ExtractFeature, FindCandidates



# add a convolutional layer to the model
def AddConvolutionalLayer(model, filter_size, kernel_size, padding, activation, normalization, input_shape=None):
    if not input_shape == None: model.add(Convolution3D(filter_size, kernel_size, padding=padding, input_shape=input_shape))
    else: model.add(Convolution3D(filter_size, kernel_size, padding=padding))

    # add activation layer
    if activation == 'LeakyReLU': model.add(LeakyReLU(alpha=0.001))
    else: model.add(Activation(activation))
    
    # add normalization after activation
    if normalization: model.add(BatchNormalization())



# add a pooling layer to the model
def AddPoolingLayer(model, pool_size, dropout, normalization):
    model.add(MaxPooling3D(pool_size=pool_size))

    # add normalization before dropout
    if normalization: model.add(BatchNormalization())

    # add dropout layer
    if dropout > 0.0: model.add(Dropout(dropout))



# add a flattening layer to the model
def AddFlattenLayer(model):
    model.add(Flatten())



# add a dense layer to the model
def AddDenseLayer(model, filter_size, dropout, activation, normalization):
    model.add(Dense(filter_size))
    if (dropout > 0.0): model.add(Dropout(dropout))

    # add activation layer
    if activation == 'LeakyReLU': model.add(LeakyReLU(alpha=0.001))
    else: model.add(Activation(activation))

    # add normalization after activation
    if normalization: model.add(BatchNormalization())




# write all relevant information to the log file
def WriteLogfiles(model, model_prefix, parameters):
    logfile = '{}.log'.format(model_prefix)

    with open(logfile, 'w') as fd:
        for layer in model.layers:
            print '{} {} -> {}'.format(layer.get_config()['name'], layer.input_shape, layer.output_shape)
            fd.write('{} {} -> {}\n'.format(layer.get_config()['name'], layer.input_shape, layer.output_shape))
        print 
        fd.write('\n')
        for parameter in parameters:
            print '{}: {}'.format(parameter, parameters[parameter])
            fd.write('{}: {}\n'.format(parameter, parameters[parameter]))



# train a neural network for this prefix
def Train(prefix, model_prefix, threshold, maximum_distance, window_radius, width, parameters):
    # identify convenient variables
    nchannels = width[3]
    starting_epoch = parameters['starting_epoch']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    initial_learning_rate = parameters['initial_learning_rate']
    decay_rate = parameters['decay_rate']

    # architecture parameters
    activation = parameters['activation']
    double_conv = parameters['double_conv']
    normalization = parameters['normalization']
    optimizer = parameters['optimizer']
    weights = parameters['weights']
    filter_size = parameters['filter_size']
    depth = parameters['depth']


    # create the model
    model = Sequential()

    # add all layers to the model
    AddConvolutionalLayer(model, filter_size, (3, 3, 3), 'valid', activation, normalization, width)
    if double_conv: AddConvolutionalLayer(model, filter_size, (3, 3, 3), 'valid', activation, normalization)
    AddPoolingLayer(model, (1, 2, 2), 0.0, normalization)

    AddConvolutionalLayer(model, 2 * filter_size, (3, 3, 3), 'valid', activation, normalization)
    if double_conv: AddConvolutionalLayer(model, 2 * filter_size, (3, 3, 3), 'valid', activation, normalization)
    AddPoolingLayer(model, (1, 2, 2), 0.0, normalization)

    if depth > 2:
        AddConvolutionalLayer(model, 4 * filter_size, (3, 3, 3), 'valid', activation, normalization)
        if double_conv: AddConvolutionalLayer(model, filter_size, (3, 3, 3), 'valid', activation, normalization)
        AddPoolingLayer(model, (2, 2, 2), 0.0, normalization)

    if depth > 3:
        AddConvolutionalLayer(model, 8 * filter_size, (3, 3, 3), 'valid', activation, normalization)
        if double_conv: AddConvolutionalLayer(model, 8 * filter_size, (3, 3, 3), 'valid', activation, normalization)
        AddPoolingLayer(model, (2, 2, 2), 0.0, normalization)

    AddFlattenLayer(model)
    AddDenseLayer(model, 512, 0.0, activation, normalization)
    AddDenseLayer(model, 1, 0.0, 'sigmoid', False)

    # compile the model
    if optimizer == 'adam': opt = Adam(lr=initial_learning_rate, decay=decay_rate, beta_1=0.99, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'sgd': opt = SGD(lr=initial_learning_rate, decay=decay_rate, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=opt)



    # make sure the folder for the model prefix exists
    root_location = model_prefix.rfind('/')
    output_folder = model_prefix[:root_location]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # write out the network parameters to a file
    WriteLogfiles(model, model_prefix, parameters)



    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    world_res = dataIO.Resolution(prefix)

    # get the radii for the relevant region
    radii = (window_radius / world_res[IB_Z], window_radius / world_res[IB_Y], window_radius / world_res[IB_X])

    # get all candidates
    candidates = FindCandidates(prefix, threshold, maximum_distance, inference=False)
    ncandidates = len(candidates)



    # determine the total number of epochs
    if parameters['augment']: rotations = 16
    else: rotations = 1

    if rotations * ncandidates % batch_size: 
        nepochs = (iterations * rotations * ncandidates / batch_size) + 1
    else:
        nepochs = (iterations * rotations * ncandidates / batch_size)



    # need to adjust learning rate and load in existing weights
    if starting_epoch == 1: index = 0
    else:
        nexamples = starting_epoch * batch_size
        current_learning_rate = initial_learning_rate / (1.0 + nexamples * decay_rate)
        backend.set_value(model.optimizer.lr, current_learning_rate)

        index = (starting_epoch * batch_size) % (ncandidates * rotations)

        model.load_weights('{}-{}.h5'.format(model_prefix, starting_epoch))



    # iterate for every epoch
    start_time = time.time()
    for epoch in range(starting_epoch, nepochs + 1):
        # print statistics
        if not epoch % 20:
            print '{}/{} in {:4f} seconds'.format(epoch, nepochs, time.time() - start_time)
            start_time = time.time()



        # create arrays for examples and labels
        examples = np.zeros((batch_size, width[IB_Z], width[IB_Y], width[IB_X], nchannels), dtype=np.uint8)
        labels = np.zeros((batch_size, 1), dtype=np.uint8)

        for iv in range(batch_size):
            # get the index and the rotation
            rotation = index / ncandidates
            candidate = candidates[index % ncandidates]

            # get the example and label
            examples[iv,:,:,:,:] = ExtractFeature(segmentation, candidate, width, radii, rotation)
            labels[iv,:] = candidate.ground_truth

            # provide overflow relief
            index += 1
            if index >= ncandidates * rotations: index = 0

        # fit the model
        model.fit(examples, labels, batch_size=batch_size, epochs=1, verbose=0, class_weight=weights)



        # save for every 1000 examples
        if not epoch % (1000 / batch_size):
            json_string = model.to_json()
            open('{}-{}.json'.format(model_prefix, epoch), 'w').write(json_string)
            model.save_weights('{}-{}.h5'.format(model_prefix, epoch))



        # update the learning rate
        nexamples = epoch * batch_size
        current_learning_rate = initial_learning_rate / (1.0 + nexamples * decay_rate)
        backend.set_value(model.optimizer.lr, current_learning_rate)



    # save the fully trained model
    json_string = model.to_json()
    open('{}.json'.format(model_prefix), 'w').write(json_string)
    model.save_weights('{}.h5'.format(model_prefix))
