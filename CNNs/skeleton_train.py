import argparse
import numpy as np
import struct
import sys
import os
from keras.optimizers import Adadelta
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution3D, MaxPooling3D
from numba import jit

# add parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utilities import dataIO
from skeleton_classifier import make_window, ReadMergeFilename


def data_generator(args, prefix):
    # read in h5 file
    filename = 'rhoana/' + prefix + '_rhoana.h5'
    segmentation = dataIO.ReadH5File(filename, 'main')

    # read in potential merge locations
    merge_filename = 'skeletons/' + prefix + '_merge_candidates.merge'
    merge_candidates, _, _ = ReadMergeFilename(merge_filename)

    num_locations = len(merge_candidates)
    batch_num = 0
    while 1:
        # create empty examples and labels arrays
        examples = np.zeros((args.batch_size, args.window_width, args.window_width, args.window_width, 3))
        labels = np.zeros((args.batch_size, 2))

        # populate the examples and labels array
        for index in range(args.batch_size):
            total = index + batch_num * args.batch_size

            # get this merge candidate
            merge_candidate = merge_candidates[total]

            # make the window given the merge candidate
            window = make_window(segmentation, merge_candidate.label_one, merge_candidate.label_two, merge_candidate.x, merge_candidate.y, merge_candidate.z, args.window_width)

            # update the input vectors
            examples[index,:,:,:,:] = window
            labels[index,:] = 1 - merge_candidate.ground_truth

        # restart the batch number if needed
        batch_num += 1
        if (batch_num + 1) * args.batch_size > num_locations:
            batch_num = 0

        # return the current examples and labels
        yield (examples, labels)

def train_network(args):
    # read in potential merge locations
    training_filename = 'skeletons/' + args.training_prefix + '_merge_candidates.merge'
    _, npositives, nnegatives = ReadMergeFilename(training_filename)

    weights = (nnegatives / float(npositives + nnegatives), npositives / float(npositives + nnegatives))

    model = Sequential()
    model.add(Convolution3D(32, 3, 3, 3, border_mode='valid',  # (49 x 49 x 49 x 32) for args.window_width = 51
                            input_shape=(args.window_width, args.window_width, args.window_width, 3)))
    model.add(Activation('relu'))
    model.add(Convolution3D(32, 3, 3, 3))  # (47 x 47 x 47 x 32)
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))  # (23 x 23 x 23 x 32)
    model.add(Dropout(0.25))

    model.add(Convolution3D(64, 3, 3, 3, border_mode='valid'))  # (21 x 21 x 21 x 64)
    model.add(Activation('relu'))
    model.add(Convolution3D(64, 3, 3, 3))  # (19 x 19 x 19 x 64)
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))  # (9 x 9 x 9 x 64)
    model.add(Dropout(0.25))

    model.add(Convolution3D(128, 3, 3, 3, border_mode='valid'))  # (7 x 7 x 7 x 128)
    model.add(Activation('relu'))
    model.add(Convolution3D(128, 3, 3, 3))  # (5 x 5 x 5 x 128)
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))  # (2 x 2 x 2 x 128)
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    adadelta = Adadelta(lr=args.learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

    model.fit_generator(data_generator(args, args.training_prefix), samples_per_epoch=args.epoch_size, verbose=1, nb_epoch=args.num_epochs,
                        validation_data=data_generator(args, args.validation_prefix), nb_val_samples=args.validation_size, class_weight=weights)

    return model

def main():
    # parse the arguments
    parser = argparse.ArgumentParser(description='Train and output a classifier for skeletons')
    parser.add_argument('training_prefix', help='Prefix for the training dataset')
    parser.add_argument('validation_prefix', help='Prefix for the validation dataset')
    parser.add_argument('output', help='Path to save the trained Keras model as an .h5 file')
    parser.add_argument('--learning_rate', default=0.5, type=float, help='Learning rate for Adadelta optimizer')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size to use during training')
    parser.add_argument('--num_epochs', default=15, type=int, help='Number of epochs in training')
    parser.add_argument('--epoch_size', default=5000, type=int, help='Number of examples per epoch.')
    parser.add_argument('--validation_size', default=2500, type=int, help='Number of examples to use for validation')
    parser.add_argument('--window_width', default=51, type=int, help='Width of window in each dimension')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='print verbosity')
    args = parser.parse_args()

    # train the model
    model = train_network(args)
    model.save(args.output)

if __name__ == '__main__':
    main()