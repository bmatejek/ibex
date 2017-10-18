import time
import numpy as np
import os


from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.cnns.skeleton.util import FindCandidates, ExtractFeature


from nn_transfer import transfer, util



###########################
#### KERAS DEFINITIONS ####
###########################

from keras.models import Model, Sequential
from keras.layers import Activation, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend



# add a convolutional layer to the model
def KerasConvolutionalLayer(model, filter_size, kernel_size, padding, activation, normalization, name, input_shape=None):
    if not input_shape == None: model.add(Convolution3D(filter_size, kernel_size, name=name, padding=padding, input_shape=input_shape))
    else: model.add(Convolution3D(filter_size, kernel_size, name=name, padding=padding))

    # add activation layer
    if activation == 'LeakyReLU': model.add(LeakyReLU(alpha=0.001))
    else: model.add(Activation(activation))
    
    # add normalization after activation
    if normalization: model.add(BatchNormalization())



# add a pooling layer to the model
def KerasPoolingLayer(model, pool_size, dropout, normalization):
    model.add(MaxPooling3D(pool_size=pool_size))

    # add normalization before dropout
    if normalization: model.add(BatchNormalization())

    # add dropout layer
    if dropout > 0.0: model.add(Dropout(dropout))



# add a flattening layer to the model
def KerasFlattenLayer(model):
    model.add(Flatten())



# add a dense layer to the model
def KerasDenseLayer(model, filter_size, dropout, activation, normalization, name):
    model.add(Dense(filter_size, name=name))
    if (dropout > 0.0): model.add(Dropout(dropout))

    # add activation layer
    if activation == 'LeakyReLU': model.add(LeakyReLU(alpha=0.001))
    else: model.add(Activation(activation))

    # add normalization after activation
    if normalization: model.add(BatchNormalization())



def KerasSkeletonNetwork(parameters, width):
    # identify convenient variables
    initial_learning_rate = parameters['initial_learning_rate']
    decay_rate = parameters['decay_rate']
    activation = parameters['activation']
    normalization = parameters['normalization']
    filter_size = parameters['filter_size']
    depth = parameters['depth']
    betas = parameters['betas']
    
    model = Sequential()

    KerasConvolutionalLayer(model, filter_size, (3, 3, 3), 'valid', activation, normalization, 'layers.0.conv1.0', width)
    KerasConvolutionalLayer(model, filter_size, (3, 3, 3), 'valid', activation, normalization, 'layers.0.conv2.0')
    KerasPoolingLayer(model, (1, 2, 2), 0.0, normalization)

    KerasConvolutionalLayer(model, 2 * filter_size, (3, 3, 3), 'valid', activation, normalization, 'layers.1.conv1.0')
    KerasConvolutionalLayer(model, 2 * filter_size, (3, 3, 3), 'valid', activation, normalization, 'layers.1.conv2.0')
    KerasPoolingLayer(model, (1, 2, 2), 0.0, normalization)

    KerasConvolutionalLayer(model, 4 * filter_size, (3, 3, 3), 'valid', activation, normalization, 'layers.2.conv1.0')
    KerasConvolutionalLayer(model, 4 * filter_size, (3, 3, 3), 'valid', activation, normalization, 'layers.2.conv2.0')
    KerasPoolingLayer(model, (2, 2, 2), 0.0, normalization)

    if depth > 3:
        KerasConvolutionalLayer(model, 8 * filter_size, (3, 3, 3), 'valid', activation, normalization, 'layers.3.conv1.0')
        KerasConvolutionalLayer(model, 8 * filter_size, (3, 3, 3), 'valid', activation, normalization, 'layers.3.conv2.0')
        KerasPoolingLayer(model, (2, 2, 2), 0.0, normalization)

    KerasFlattenLayer(model)
    KerasDenseLayer(model, 512, 0.0, activation, normalization, 'layers.4.fc.0')
    KerasDenseLayer(model, 1, 0.0, 'sigmoid', False, 'layers.5.fc.0')

    optimizer = Adam(lr=initial_learning_rate, decay=decay_rate, beta_1=betas[0], beta_2=betas[1], epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model
    



###########################
#### TORCH DEFINITIONS ####
###########################

import torch
import torch.nn as nn



class TorchConvolutionLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=(3,3,3), pool_size=(1,2,2), normalization=False, activation='LeakyReLU'):
        # call parent constructor
        super(TorchConvolutionLayer, self).__init__()

        if activation == 'LeakyReLU': activation_layer = nn.LeakyReLU(0.001)
        elif activation == 'relu': activation_layer = nn.ReLU()
        else: sys.stderr('Unrecognized activation {}'.format(activation))
        
        if normalization:
            self.conv1 = nn.Sequential(nn.Conv3d(input_size[0], output_size[0], kernel_size=kernel_size), nn.BatchNorm3d(output_size[0]), activation_layer)
            self.conv2 = nn.Sequential(nn.Conv3d(input_size[1], output_size[1], kernel_size=kernel_size), nn.BatchNorm3d(output_size[1]), activation_layer)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(input_size[0], output_size[0], kernel_size=kernel_size), activation_layer)
            self.conv2 = nn.Sequential(nn.Conv3d(input_size[1], output_size[1], kernel_size=kernel_size), activation_layer)
        self.down = nn.MaxPool3d(pool_size)

    def forward(self, inputs):
        return self.down(self.conv2(self.conv1(inputs)))


class TorchFlattenLayer(nn.Module):
    def __init__(self):
        # call parent constructor
        super(TorchFlattenLayer, self).__init__()

    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)



class TorchDenseLayer(nn.Module):
    def __init__(self, input_size, output_size, normalization, activation):
        # call the parent constructor
        super(TorchDenseLayer, self).__init__()

        if activation == 'LeakyReLU': activation_layer = nn.LeakyReLU(0.001)
        elif activation == 'relu': activation_layer = nn.ReLU()
        elif activation == 'sigmoid': activation_layer = nn.Sigmoid()
        else: sys.stderr('Unrecognized activation {}'.format(activation))
        
        if normalization: 
            self.fc = nn.Sequential(nn.Linear(input_size, output_size), nn.BatchNorm1d(output_size), activation_layer)
        else:
            self.fc = nn.Sequential(nn.Linear(input_size, output_size), activation_layer)

            
    def forward(self, inputs):
        return self.fc(inputs)



class TorchSkeletonNetwork(nn.Module):
    def __init__(self, parameters):
        # call parent constructor
        super(TorchSkeletonNetwork, self).__init__()

        # save useful instance variables
        normalization = parameters['normalization']
        output_width = parameters['output_width']
        filter_size = parameters['filter_size']
        activation = parameters['activation']
        
        # get the kernel and pooling sizes
        kernel_sizes = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        pooling_sizes = [(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)]

        # add all of the convolution layers
        if parameters['depth'] == 3:
            self.layers = nn.ModuleList([
                TorchConvolutionLayer([3, filter_size], [filter_size, filter_size], kernel_sizes[0], pooling_sizes[0], normalization, activation),
                TorchConvolutionLayer([filter_size, 2 * filter_size], [2 * filter_size, 2 * filter_size], kernel_sizes[1], pooling_sizes[1], normalization, activation),
                TorchConvolutionLayer([2 * filter_size, 4 * filter_size], [4 * filter_size, 4 * filter_size], kernel_sizes[2], pooling_sizes[2], normalization, activation),
                TorchFlattenLayer(),
                TorchDenseLayer(4 * filter_size * output_width * output_width * output_width, 512, normalization, activation),
                TorchDenseLayer(512, 1, normalization, 'sigmoid')
            ])
        else:
            self.layers = nn.ModuleList([
                TorchConvolutionLayer([3, filter_size], [filter_size, filter_size], kernel_sizes[0], pooling_sizes[0], normalization, activation),
                TorchConvolutionLayer([filter_size, 2 * filter_size], [2 * filter_size, 2 * filter_size], kernel_sizes[1], pooling_sizes[1], normalization, activation),
                TorchConvolutionLayer([2 * filter_size,4 * filter_size], [4 * filter_size, 4 * filter_size], kernel_sizes[2], pooling_sizes[2], normalization, activation),
                TorchConvolutionLayer([4 * filter_size,8 * filter_size], [8 * filter_size, 8 * filter_size], kernel_sizes[3], pooling_sizes[3], normalization, activation),
                TorchFlattenLayer(),
                TorchDenseLayer(8 * filter_size * output_width * output_width * output_width, 512, normalization, activation),
                TorchDenseLayer(512, 1, normalization, 'sigmoid')
            ])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs



def VerifyWeights(pytorch_model, keras_model):
    keras_conv1 = keras_model.layers[0].get_weights()[0]
    keras_conv2 = keras_model.layers[2].get_weights()[0]
    keras_conv3 = keras_model.layers[5].get_weights()[0]
    keras_conv4 = keras_model.layers[7].get_weights()[0]
    keras_conv5 = keras_model.layers[10].get_weights()[0]
    keras_conv6 = keras_model.layers[12].get_weights()[0]
    keras_dense1 = keras_model.layers[16].get_weights()[0]
    keras_dense2 = keras_model.layers[18].get_weights()[0]

    # print the pytorch layer
    if False:
        for iv in range(3):
            for iy in range(2):
                print pytorch_model.layers[iv].conv1[iy]
            for iy in range(2):
                print pytorch_model.layers[iv].conv2[iy]
            print pytorch_model.layers[iv].down
        print pytorch_model.layers[3]
        for iv in range(4, 6):
            for iy in range(2):
                print pytorch_model.layers[iv].fc[iy]
        print keras_model.summary()

            
            
    pytorch_conv1 = pytorch_model.layers[0].conv1[0].weight.data.cpu().numpy()
    pytorch_conv2 = pytorch_model.layers[0].conv2[0].weight.data.cpu().numpy()
    pytorch_conv3 = pytorch_model.layers[1].conv1[0].weight.data.cpu().numpy()
    pytorch_conv4 = pytorch_model.layers[1].conv2[0].weight.data.cpu().numpy()
    pytorch_conv5 = pytorch_model.layers[2].conv1[0].weight.data.cpu().numpy()
    pytorch_conv6 = pytorch_model.layers[2].conv2[0].weight.data.cpu().numpy()
    pytorch_dense1 = pytorch_model.layers[4].fc[0].weight.data.cpu().numpy()
    pytorch_dense2 = pytorch_model.layers[5].fc[0].weight.data.cpu().numpy()
    
    #print np.mean(abs(np.transpose(pytorch_conv1, (2, 3, 4, 1, 0)) - keras_conv1))
    #print np.mean(abs(np.transpose(pytorch_conv2, (2, 3, 4, 1, 0)) - keras_conv2))
    #print np.mean(abs(np.transpose(pytorch_conv3, (2, 3, 4, 1, 0)) - keras_conv3))
    #print np.mean(abs(np.transpose(pytorch_conv4, (2, 3, 4, 1, 0)) - keras_conv4))
    #print np.mean(abs(np.transpose(pytorch_conv5, (2, 3, 4, 1, 0)) - keras_conv5))
    #print np.mean(abs(np.transpose(pytorch_conv6, (2, 3, 4, 1, 0)) - keras_conv6))
    assert (np.mean(abs(np.transpose(pytorch_dense1) - keras_dense1)) < 1e-6)
    assert (np.mean(abs(np.transpose(pytorch_dense2) - keras_dense2)) < 1e-6)
    


def Train(prefix, model_prefix, threshold, maximum_distance, network_distance, width, parameters):
    # identify convenient variables
    nchannels = width[0]
    starting_epoch = parameters['starting_epoch']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    initial_learning_rate = parameters['initial_learning_rate']
    decay_rate = parameters['decay_rate']

    # architecture parameters
    activation = parameters['activation']
    weights = parameters['weights']
    betas = parameters['betas']


    ###############################
    #### SET UP OUTPUT ENVIRON ####
    ###############################
    
    # make sure the folder for the model prefix exists
    root_location = model_prefix.rfind('/')
    output_folder = model_prefix[:root_location]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # open up the log file with no buffer
    logfile = open('{}.log'.format(model_prefix), 'w', 0) 



    #######################
    #### SET UP MODELS ####
    #######################
    
    # set up the keras model
    keras_model = KerasSkeletonNetwork(parameters, width)
    
    # set up the pytorch model model
    pytorch_model = TorchSkeletonNetwork(parameters)

    # transfer the weights from keras to pytorch
    transfer.keras_to_pytorch(keras_model, pytorch_model, flip_filters=True)

    # set the parameters for pytorch
    pytorch_model.cuda()
    pytorch_model.train()
    # get the optimizer and loss function
    optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=initial_learning_rate, weight_decay=decay_rate, betas=betas, eps=1e-08)
    loss_function = torch.nn.MSELoss()

    # create torch arrays for training
    x = torch.autograd.Variable(torch.zeros(batch_size, nchannels, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1]).cuda(), requires_grad=False)
    y = torch.autograd.Variable(torch.zeros(batch_size).cuda(), requires_grad=False)



    ###########################################
    #### SET UP GENERAL SKELETON VARIABLES ####
    ###########################################

    # read in all relevant information
    segmentation = dataIO.ReadSegmentationData(prefix)
    world_res = dataIO.Resolution(prefix)

    # get the radii for the relevant region
    radii = (network_distance / world_res[IB_Z], network_distance / world_res[IB_Y], network_distance / world_res[IB_X])

    # get all candidates
    candidates = FindCandidates(prefix, threshold, maximum_distance, network_distance, inference=False)
    ncandidates = len(candidates)

    # determine the total number of epochs

    if parameters['augment']: rotations = 16
    else: rotations = 1

    if rotations * ncandidates % batch_size: 
        nepochs = (iterations * rotations * ncandidates / batch_size) + 1
    else:
        nepochs = (iterations * rotations * ncandidates / batch_size)



    # need to adjust learning rate and load in existing weights
    if starting_epoch == 1:
        index = 0
    else:
        print 'Starting epoch feature not implented yet'

    
    # iterate for every epoch
    cumulative_time = time.time()
    for epoch in range(starting_epoch, nepochs + 1):
        transfer.keras_to_pytorch(keras_model, pytorch_model, flip_filters=True, verbose=False)
        VerifyWeights(pytorch_model, keras_model)

        # update the learning rate
        nexamples = (epoch - 1) * batch_size
        current_learning_rate = initial_learning_rate / (1.0 + nexamples * decay_rate)
        optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=current_learning_rate, weight_decay=decay_rate, betas=betas, eps=1e-08)
        backend.set_value(keras_model.optimizer.lr, current_learning_rate)

        
        start_time = time.time()

        # set gradient to zero
        optimizer.zero_grad()

        # create arrays for examples and labels
        examples = np.zeros((batch_size, nchannels, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1]), dtype=np.uint8)
        labels = np.zeros(batch_size, dtype=np.uint8)

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
            
        #### RUN FORWARD ITERATION FOR TORCH ####
        x.data.copy_(torch.from_numpy(examples))
        y.data.copy_(torch.from_numpy(labels))
        
        y_prediction = pytorch_model(x)
        loss = loss_function(y_prediction, y)
        loss.backward()
        optimizer.step()

        # print verbosity
        pytorch_time = time.time()
        print 'TORCH [Iter {} / {}] loss = {:.7f} Total Time = {:.2f} seconds'.format(epoch, nepochs, loss.data[0], pytorch_time - start_time)
        
        
        
        #### RUN FORWARD ITERATION FOR KERAS ###
        history = keras_model.fit(examples, labels, batch_size=batch_size, epochs=1, verbose=0, class_weight=weights, shuffle=False)

        # print verbosity
        keras_time = time.time()
        print 'KERAS [Iter {} / {}] loss = {:.7f} Total Time = {:.2f} seconds'.format(epoch, nepochs, history.history['loss'][0], keras_time - start_time)
        
        #pytorch_prediction = pytorch_model.layers[0].conv1[0](x).data.cpu().numpy()
        #model = Model(input=keras_model.inputs, output=keras_model.layers[0].output)
        #keras_prediction = model.predict(examples)

        if not epoch % 1000:
            torch.save({'epoch': epoch, 'state_dict': pytorch_model.state_dict(), 'optmizer': optimizer.state_dict()}, '{}-{}.arch'.format(model_prefix, epoch))
            
        assert (abs(loss.data[0] - history.history['loss'][0]) < 10e-4)

    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optmizer': optimizer.state_dict()}, '{}.arch'.format(model_prefix))


        
