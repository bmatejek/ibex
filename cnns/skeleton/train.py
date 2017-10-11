import numpy as np
import time
import os

import torch
import torch.nn as nn


from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.cnns.skeleton.util import FindCandidates, ExtractFeature



class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=(3,3,3), pool_size=(1,2,2), normalization=False):
        # call parent constructor
        super(ConvolutionLayer, self).__init__()
        if normalization:
            self.conv1 = nn.Sequential(nn.Conv3d(input_size[0], output_size[0], kernel_size=kernel_size), nn.BatchNorm3d(output_size[0]), nn.LeakyReLU(0.001))
            self.conv2 = nn.Sequential(nn.Conv3d(input_size[1], output_size[1], kernel_size=kernel_size), nn.BatchNorm3d(output_size[1]), nn.LeakyReLU(0.001))
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(input_size[0], output_size[0], kernel_size=kernel_size), nn.LeakyReLU(0.001))
            self.conv2 = nn.Sequential(nn.Conv3d(input_size[1], output_size[1], kernel_size=kernel_size), nn.LeakyReLU(0.001))
        self.down = nn.MaxPool3d(pool_size)

    def forward(self, inputs):
        return self.down(self.conv2(self.conv1(inputs)))


class FlattenLayer(nn.Module):
    def __init__(self):
        # call parent constructor
        super(FlattenLayer, self).__init__()

    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)



class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size, normalization):
        # call the parent constructor
        super(DenseLayer, self).__init__()

        if normalization: 
            self.fc = nn.Sequential(nn.Linear(input_size, output_size), nn.BatchNorm3d(output_size), nn.LeakyReLU(0.001))
        else:
            self.fc = nn.Sequential(nn.Linear(input_size, output_size), nn.LeakyReLU(0.001))

    def forward(self, inputs):
        return self.fc(inputs)


class FinalLayer(nn.Module):
    def __init__(self):
        # call the parent constructor
        super(FinalLayer, self).__init__()

    def forward(self, inputs):
        return nn.functional.sigmoid(inputs)


class SkeletonNetwork(nn.Module):
    def __init__(self, parameters, width):
        # call parent constructor
        super(SkeletonNetwork, self).__init__()

        # save useful instance variables
        normalization = parameters['normalization']
        output_width = parameters['output_width']
        filter_size = parameters['filter_size']

        # get the kernel and pooling sizes
        kernel_sizes = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        pooling_sizes = [(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)]

        # add all of the convolution layers
        if parameters['depth'] == 3:
            self.layers = nn.ModuleList([
                ConvolutionLayer([3, filter_size], [filter_size, filter_size], kernel_sizes[0], pooling_sizes[0], normalization),
                ConvolutionLayer([filter_size, 2 * filter_size], [2 * filter_size, 2 * filter_size], kernel_sizes[1], pooling_sizes[1], normalization),
                ConvolutionLayer([2 * filter_size, 4 * filter_size], [4 * filter_size, 4 * filter_size], kernel_sizes[2], pooling_sizes[2], normalization),
                FlattenLayer(),
                DenseLayer(4 * filter_size * output_width * output_width * output_width, 512, normalization),
                DenseLayer(512, 1, normalization),
                FinalLayer()
            ])
        else:
            self.layers = nn.ModuleList([
                ConvolutionLayer([3, filter_size], [filter_size, filter_size], kernel_sizes[0], pooling_sizes[0], normalization),
                ConvolutionLayer([filter_size, 2 * filter_size], [2 * filter_size, 2 * filter_size], kernel_sizes[1], pooling_sizes[1], normalization),
                ConvolutionLayer([2 * filter_size,4 * filter_size], [4 * filter_size, 4 * filter_size], kernel_sizes[2], pooling_sizes[2], normalization),
                ConvolutionLayer([4 * filter_size,8 * filter_size], [8 * filter_size, 8 * filter_size], kernel_sizes[3], pooling_sizes[3], normalization),
                FlattenLayer(),
                DenseLayer(8 * filter_size * output_width * output_width * output_width, 512, normalization),
                DenseLayer(512, 1, normalization),
                FinalLayer()
            ])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs



def Train(prefix, model_prefix, threshold, maximum_distance, network_distance, width, parameters):
    # identify convenient variables
    nchannels = width[3]
    starting_epoch = parameters['starting_epoch']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    initial_learning_rate = parameters['initial_learning_rate']
    decay_rate = parameters['decay_rate']

    # architecture parameters
    activation = parameters['activation']
    weights = parameters['weights']
    betas = parameters['betas']
    
    # set up model
    model = SkeletonNetwork(parameters, width)
    model.cuda()
    model.train()

    # get the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, betas=betas, weight_decay=decay_rate)
    loss_function = torch.nn.MSELoss()



    # make sure the folder for the model prefix exists
    root_location = model_prefix.rfind('/')
    output_folder = model_prefix[:root_location]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # open up the log file with no buffer
    logfile = open('{}.log'.format(model_prefix), 'w', 0) 



    # create torch arrays for training
    x = torch.autograd.Variable(torch.zeros(batch_size, nchannels, width[IB_Z], width[IB_Y], width[IB_X]).cuda(), requires_grad=True)
    y = torch.autograd.Variable(torch.zeros(batch_size).cuda(), requires_grad=False)




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
    # set the index to 0
    index = 0

    # iterate for every epoch
    cumulative_time = time.time()
    for epoch in range(starting_epoch, nepochs + 1):
        start_time = time.time()

        # set gradient to zero
        optimizer.zero_grad()

        # create arrays for examples and labels
        examples = np.zeros((batch_size, nchannels, width[IB_Z], width[IB_Y], width[IB_X]), dtype=np.uint8)
        labels = np.zeros(batch_size, dtype=np.uint8)

        for iv in range(batch_size):
            # get the index and the rotation
            rotation = index / ncandidates
            candidate = candidates[index % ncandidates]

            # get the example and label
            examples[iv,:,:,:,:] = ExtractFeature(segmentation, candidate, width, radii, rotation)
            labels[iv] = candidate.ground_truth

            # provide overflow relief
            index += 1
            if index >= ncandidates * rotations: index = 0


        # copy data to torch array
        x.data.copy_(torch.from_numpy(examples))
        y.data.copy_(torch.from_numpy(labels))
        
        forward_time = time.time()
        y_prediction = model(x)
        loss = loss_function(y_prediction, y)
        loss.backward()
        optimizer.step()

        # print log
        end_time = time.time()
        logfile.write('[Iter {} / {}] loss={} Model Time = {} Total Time = {}\n'.format(epoch, nepochs, loss.data[0], end_time - forward_time, end_time - start_time))
        if not epoch % 100: 
            print '[Iter {} / {}] loss={} Total Time = {}'.format(epoch, nepochs, loss.data[0], time.time() - cumulative_time)
            cumulative_time = time.time()
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, '{}-{}.arch'.format(model_prefix, epoch))


    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optmizer': optimizer.state_dict()}, '{}.arch'.format(model_prefix, epoch))
    logfile.close()
