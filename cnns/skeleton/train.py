import numpy as np
import os
import sys
import math


from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.cnns.skeleton.util import FindCandidates, ExtractFeature


from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
import keras


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# add a convolutional layer to the model
def ConvolutionalLayer(model, filter_size, kernel_size, padding, activation, normalization, input_shape=None):
    if not input_shape == None: model.add(Convolution3D(filter_size, kernel_size, padding=padding, input_shape=input_shape))
    else: model.add(Convolution3D(filter_size, kernel_size, padding=padding))

    # add activation layer
    if activation == 'LeakyReLU': model.add(LeakyReLU(alpha=0.001))
    else: model.add(Activation(activation))
    
    # add normalization after activation
    if normalization: model.add(BatchNormalization())



# add a pooling layer to the model
def PoolingLayer(model, pool_size, dropout, normalization):
    model.add(MaxPooling3D(pool_size=pool_size))

    # add normalization before dropout
    if normalization: model.add(BatchNormalization())

    # add dropout layer
    if dropout > 0.0: model.add(Dropout(dropout))



# add a flattening layer to the model
def FlattenLayer(model):
    model.add(Flatten())



# add a dense layer to the model
def DenseLayer(model, filter_size, dropout, activation, normalization):
    model.add(Dense(filter_size))
    if (dropout > 0.0): model.add(Dropout(dropout))

    # add activation layer
    if activation == 'LeakyReLU': model.add(LeakyReLU(alpha=0.001))
    else: model.add(Activation(activation))

    # add normalization after activation
    if normalization: model.add(BatchNormalization())


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, model_prefix):
        super(PlotLosses, self).__init__()
        self.model_prefix = model_prefix

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i = self.i + 1
        
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        plt.savefig('{}-training-curve.png'.format(self.model_prefix))
        plt.gcf().clear()


def SkeletonNetwork(parameters, width):
    # identify convenient variables
    initial_learning_rate = parameters['initial_learning_rate']
    decay_rate = parameters['decay_rate']
    activation = parameters['activation']
    normalization = parameters['normalization']
    filter_sizes = parameters['filter_sizes']
    depth = parameters['depth']
    optimizer = parameters['optimizer']
    betas = parameters['betas']
    loss_function = parameters['loss_function']
    assert (len(filter_sizes) >= depth)

    model = Sequential()

    ConvolutionalLayer(model, filter_sizes[0], (3, 3, 3), 'valid', activation, normalization, width)
    ConvolutionalLayer(model, filter_sizes[0], (3, 3, 3), 'valid', activation, normalization)
    PoolingLayer(model, (1, 2, 2), 0.2, normalization)

    ConvolutionalLayer(model, filter_sizes[1], (3, 3, 3), 'valid', activation, normalization)
    ConvolutionalLayer(model, filter_sizes[1], (3, 3, 3), 'valid', activation, normalization)
    PoolingLayer(model, (1, 2, 2), 0.2, normalization)

    ConvolutionalLayer(model, filter_sizes[2], (3, 3, 3), 'valid', activation, normalization)
    ConvolutionalLayer(model, filter_sizes[2], (3, 3, 3), 'valid', activation, normalization)
    PoolingLayer(model, (2, 2, 2), 0.2, normalization)

    if depth > 3:
        ConvolutionalLayer(model, filter_sizes[3], (3, 3, 3), 'valid', activation, normalization)
        ConvolutionalLayer(model, filter_sizes[3], (3, 3, 3), 'valid', activation, normalization)
        PoolingLayer(model, (2, 2, 2), 0.2, normalization)

    FlattenLayer(model)
    DenseLayer(model, 512, 0.2, activation, normalization)
    DenseLayer(model, 1, 0.5, 'sigmoid', False)

    if optimizer == 'adam': opt = Adam(lr=initial_learning_rate, decay=decay_rate, beta_1=betas[0], beta_2=betas[1], epsilon=1e-08)
    elif optimizer == 'nesterov': opt = SGD(lr=initial_learning_rate, decay=decay_rate, momentum=0.9, nesterov=True)
    model.compile(loss=loss_function, optimizer=opt, metrics=['mean_squared_error', 'accuracy'])
    
    return model



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


def SkeletonCandidateGenerator(prefix, network_distance, candidates, parameters, width):
    # get the number of channels for the data
    nchannels = width[0]

    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    world_res = dataIO.Resolution(prefix)

    # get the radii for the relevant region
    radii = (network_distance / world_res[IB_Z], network_distance / world_res[IB_Y], network_distance / world_res[IB_X])

    # determine the total number of epochs
    if parameters['augment']: rotations = 16
    else: rotations = 1

    ncandidates = len(candidates)
    batch_size = parameters['batch_size']
    if rotations * ncandidates % batch_size: 
        nbatches = (rotations * ncandidates / batch_size) + 1
    else:
        nbatches = (rotations * ncandidates / batch_size)

    examples = np.zeros((batch_size, nchannels, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1]), dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.float32)
    
    while True:
        index = 0
        for _ in range(nbatches):
            for iv in range(batch_size):
                # get the candidate index and the rotation
                rotation = index / ncandidates
                candidate = candidates[index % ncandidates]

                # get the example and label
                examples[iv,:,:,:,:] = ExtractFeature(segmentation, candidate, width, radii, rotation)
                labels[iv] = candidate.ground_truth

                # provide overflow relief
                index += 1
                if index >= ncandidates * rotations: index = 0
                
            yield (examples, labels)



def Train(prefix, model_prefix, threshold, maximum_distance, network_distance, width, parameters):
    # identify convenient variables
    nchannels = width[0]
    starting_epoch = parameters['starting_epoch']
    batch_size = parameters['batch_size']
    initial_learning_rate = parameters['initial_learning_rate']
    decay_rate = parameters['decay_rate']

    # architecture parameters
    activation = parameters['activation']
    weights = parameters['weights']
    betas = parameters['betas']
    
    # set up the keras model
    model = SkeletonNetwork(parameters, width)
    
    # make sure the folder for the model prefix exists
    root_location = model_prefix.rfind('/')
    output_folder = model_prefix[:root_location]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)



    # open up the log file with no buffer
    logfile = '{}.log'.format(model_prefix)

    # if the file exists do not continue
    if os.path.isfile(logfile) and starting_epoch == 1:
        sys.stderr.write('Discovered {}, exiting'.format(logfile))
        sys.exit()

    # write out the network parameters to a file
    WriteLogfiles(model, model_prefix, parameters)

    # get all candidates
    training_candidates = FindCandidates(prefix, threshold, maximum_distance, network_distance, inference=False, validation=False)
    ntraining_candidates = len(training_candidates)
    validation_candidates = FindCandidates(prefix, threshold, maximum_distance, network_distance, inference=False, validation=True)
    nvalidation_candidates = len(validation_candidates)

    # create a set of keras callbacks
    callbacks = []
    
    # stop if patience number of epochs does not improve result
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    callbacks.append(early_stopping)
    
    # save the best model seen so far
    best_loss = keras.callbacks.ModelCheckpoint('{}-best-loss.h5'.format(model_prefix), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    callbacks.append(best_loss)
    best_acc = keras.callbacks.ModelCheckpoint('{}-best-acc.h5'.format(model_prefix), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    callbacks.append(best_acc)
    all_models = keras.callbacks.ModelCheckpoint(model_prefix + '-{epoch:03d}.h5', verbose=0, save_best_only=False, save_weights_only=True)
    callbacks.append(all_models)

    # plot the loss functions
    plot_losses = PlotLosses(model_prefix)
    callbacks.append(plot_losses)

    # determine the total number of epochs
    if parameters['augment']: rotations = 16
    else: rotations = 1

    # save the json file
    json_string = model.to_json()
    open('{}.json'.format(model_prefix), 'w').write(json_string)
    
    if not starting_epoch == 1:
        model.load_weights('{}-{:03d}.h5'.format(model_prefix, starting_epoch))

    history = model.fit_generator(SkeletonCandidateGenerator(prefix, network_distance, training_candidates, parameters, width),\
                    (rotations * ntraining_candidates / batch_size), epochs=500, verbose=1, class_weight=weights, callbacks=callbacks,\
                    validation_data=SkeletonCandidateGenerator(prefix, network_distance, validation_candidates, parameters, width), validation_steps=(rotations * nvalidation_candidates / batch_size))

    # save the fully trained model
    model.save_weights('{}.h5'.format(model_prefix))
