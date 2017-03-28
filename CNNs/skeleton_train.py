import argparse
import numpy as np
import struct
import sys
import os
from keras.optimizers import Adadelta, Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution3D, MaxPooling3D
from numba import jit
from keras import backend as K
#
# add parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utilities import dataIO
from skeleton_classifier import make_window, ReadMergeFilename, apply_rotation

def maybe_print(tensor, msg, do_print=False):
    if do_print:
        return K.print_tensor(tensor, msg)
    else:
        return tensor

def weighted_mse(y_true, y_pred):
    epsilon=0.00001
    y_pred = K.clip(y_pred,epsilon, 1-epsilon)
    # per batch positive fraction, negative fraction (0.5 = ignore)
    pos_mask = K.cast(y_true > 0.75, 'float32')
    neg_mask = K.cast(y_true < 0.25, 'float32')
    num_pixels = K.cast(K.prod(K.shape(y_true)[1:]), 'float32')
    pos_fracs = K.clip((K.sum(pos_mask)/num_pixels),0.01, 0.99)
    neg_fracs = K.clip((K.sum(neg_mask) /num_pixels),0.01, 0.99)

    pos_fracs = maybe_print(pos_fracs, "positive fraction",do_print=False)

    # chosen to sum to 1 when multiplied by their fractions, assuming no ignore
    pos_weight = maybe_print(1.0 / (2 * pos_fracs), "positive weight")
    neg_weight = maybe_print(1.0 / (2 * neg_fracs), "negative weight")

    per_pixel_weights = pos_weight * pos_mask + neg_weight * neg_mask
    per_pixel_weighted_sq_error = K.square(y_true - y_pred) * per_pixel_weights

    batch_weighted_mse = K.mean(per_pixel_weighted_sq_error)/2.0

    return K.mean(batch_weighted_mse)

def data_generator(args, prefix, forward=False):
    # read in h5 file
    filename = 'rhoana/' + prefix + '_rhoana.h5'
    segmentation = dataIO.ReadH5File(filename, 'main')

    # read in potential merge locations
    # TODO remove hardcoding
    if forward: merge_filename = 'skeletons/' + prefix + '_merge_candidates_forward_400nm.merge'
    else: merge_filename = 'skeletons/' + prefix + '_merge_candidates_train_400nm.merge'
    merge_candidates, _, _, radii = ReadMergeFilename(merge_filename)

    num_locations = len(merge_candidates)
    batch_num = 0

    while 1:
        # create empty examples and labels arrays
        examples = np.zeros((args.batch_size, args.window_width, args.window_width, args.window_width, 1))
        labels = np.zeros((args.batch_size, 2))

        # populate the examples and labels array
        for index in range(args.batch_size):
            total = index + batch_num * args.batch_size

            # get this merge candidate
            merge_candidate = merge_candidates[total]

            # get the labels for this candidate
            label_one = merge_candidate.label_one
            label_two = merge_candidate.label_two

            # get the position for this candidate
            xposition = merge_candidate.x
            yposition = merge_candidate.y
            zposition = merge_candidate.z

            # get the rotation
            rotation = merge_candidate.rotation

            window = make_window(segmentation, label_one, label_two, xposition, yposition, zposition, radii, args.window_width)

            # update the input vectors
            examples[index,:,:,:,:] = apply_rotation(window, rotation)
            labels[index,:] = 1 - merge_candidate.ground_truth

        # restart the batch number if needed
        batch_num += 1
        if (batch_num + 1) * args.batch_size > num_locations:
            batch_num = 0

        # return the current examples and labels
        yield (examples, labels)



# create a convolutional layer followed by an activation layer
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



# add a 3d pooling layer
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



# add a flattening layer
def AddFlattenLayer(model):
    # get the current level
    level = len(model.layers)
    # flatten the model
    model.add(Flatten())
    # print out input and output shape
    print 'Flatten Layer: ' + str(model.layers[level].input_shape) + ' -> ' + str(model.layers[level].output_shape)



# add a dense layer
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



def train_network(args):
    # read in potential merge locations
    # TODO fix hardcoding of nanometers
    training_filename = 'skeletons/' + args.training_prefix + '_merge_candidates_train_400nm.merge'
    _, npositives, nnegatives, _ = ReadMergeFilename(training_filename)

    weights = (npositives / float(npositives + nnegatives), nnegatives / float(npositives + nnegatives))

    model = Sequential()

    AddConvolutionLayer(model, 32, (3, 3, 3), padding='valid', activation='relu', input_shape=(args.window_width, args.window_width, args.window_width, 1))
    AddConvolutionLayer(model, 32, (3, 3, 3), padding='valid', activation='relu')
    AddPoolingLayer(model, (2, 2, 2), dropout=0.25)

    # AddConvolutionLayer(model, 32, (3, 3, 3), padding='valid', activation='relu')
    # AddConvolutionLayer(model, 32, (3, 3, 3), padding='valid', activation='relu')
    # AddPoolingLayer(model, (2, 2, 2), dropout=0.25)

    AddConvolutionLayer(model, 64, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 64, (3, 3, 3), padding='valid', activation='relu')
    AddPoolingLayer(model, (2, 2, 2), dropout=0.25)
    
    AddConvolutionLayer(model, 128, (3, 3, 3), padding='valid', activation='relu')
    AddConvolutionLayer(model, 128, (3, 3, 3), padding='valid', activation='relu')
    AddPoolingLayer(model, (2, 2, 2), dropout=0.25)

    AddFlattenLayer(model)
    AddDenseLayer(model, 512, dropout=0.5, activation='relu')
    AddDenseLayer(model, 2, dropout=0.0, activation='sigmoid')

    #adadelta = Adadelta(lr=args.learning_rate)
    adm = Adam(lr=1e-4, beta_1=0.99, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=weighted_mse, optimizer=adm)

    model.fit_generator(data_generator(args, args.training_prefix), steps_per_epoch=args.epoch_size, verbose=2, epochs=args.num_epochs, class_weight=weights)

    return model

def main():
    # parse the arguments
    parser = argparse.ArgumentParser(description='Train and output a classifier for skeletons')
    parser.add_argument('training_prefix', help='Prefix for the training dataset')
    parser.add_argument('validation_prefix', help='Prefix for the validation dataset')
    parser.add_argument('output', help='Path to save the trained Keras model as an .h5 file')
    #parser.add_argument('--learning_rate', default=0.5, type=float, help='Learning rate for Adadelta optimizer')
    parser.add_argument('--batch_size', default=5, type=int, help='Batch size to use during training')
    parser.add_argument('--num_epochs', default=2000, type=int, help='Number of epochs in training')
    parser.add_argument('--epoch_size', default=5, type=int, help='Number of examples per epoch.')
    parser.add_argument('--window_width', default=51, type=int, help='Width of window in each dimension')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='print verbosity')
    args = parser.parse_args()

    # train the model
    model = train_network(args)

    json_string = model.to_json()
    open(args.output.replace('h5', 'json'), 'w').write(json_string)
    model.save_weights(args.output)

if __name__ == '__main__':
    main()