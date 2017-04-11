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
def Train(prefix, maximum_distance, output_prefix, num_epochs=-1, window_width=106):
    # read in the h5 segmentation file
    segmentation = dataIO.ReadSegmentationData(prefix)

    # get the grid size and the world resolution in (z, y, x)
    grid_size = segmentation.shape
    world_res = dataIO.ReadMetaData(prefix)

    # get the radii for the bounding box in grid coordinates
    radii = (maximum_distance / world_res[0], maximum_distance / world_res[1], maximum_distance / world_res[2])

    # get all of the candidates for this prefix
    candidates = FindCandidates(prefix, maximum_distance, forward=False)
    ncandidates = len(candidates)

    # the number of rotations for every example
    nrotations = 8

    # create the model
    model = Sequential()

    AddConvolutionLayer(model, 16, (3, 3, 3), padding='valid', activation='relu', input_shape=(window_width, window_width, window_width, 1))
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
        num_epochs = nrotations * ncandidates / 2

    # keep track of the number of epochs here separately
    # this may reset to 0 if number of examples reached
    index = 0

    # run for all epochs and time for every group of 20
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        if not epoch % 20:
            print '{0}/{1} in {2:4f} seconds'.format(epoch, num_epochs, time.time() - start_time)
            start_time = time.time()

        # get the index and the rotation
        positive_candidate_rotation = index / ncandidates
        positive_candidate_index = index % ncandidates
        
        # receive the candidate
        positive_candidate = candidates[positive_candidate_index]

        # get the information about this candidate
        positive_labels = positive_candidate.Labels()
        positive_location = positive_candidate.Location()

        # get the example for this candidate and rotation
        positive_example = ExtractFeature(segmentation, positive_labels, positive_location, radii, window_width, positive_candidate_rotation)

        # increment the index
        index += 1

        # get the index and the rotation
        negative_candidate_rotation = index / ncandidates
        negative_candidate_index = index % ncandidates

        # receive the candidate
        negative_candidate = candidates[negative_candidate_index]

        # get the information about this candidate
        negative_labels = negative_candidate.Labels()
        negative_location = negative_candidate.Location()

        # get the example for this candidate rotation
        negative_example = ExtractFeature(segmentation, negative_labels, negative_location, radii, window_width, negative_candidate_rotation)

        # increment the index
        index += 1

        # merge the two examples together
        examples = np.zeros((2, window_width, window_width, window_width, 1))
        examples[0,:,:,:,:] = positive_example
        examples[1,:,:,:,:] = negative_example

        # create the label for this example
        labels = np.zeros((2, 1))
        labels[0,:] = positive_candidate.GroundTruth()
        labels[1,:] = negative_candidate.GroundTruth()
        assert (labels[0,0] != labels[1,0])

        # update the learning rate
        current_learning_rate = initial_learning_rate / (1.0 + (epoch - 1) * decay_rate)
        backend.set_value(model.optimizer.lr, current_learning_rate)

        # fit the model
        model.fit(examples, labels, epochs=1, verbose=0)

        if not epoch % 500:
            # save an indermediate model
            json_string = model.to_json()
            open(output_prefix + '-' + str(epoch) + '.json', 'w').write(json_string)
            model.save_weights(output_prefix + '-' + str(epoch) + '.h5')

    # save the fully trained model
    json_string = model.to_json()
    open(output_prefix + '.json', 'w').write(json_string)
    model.save_weights(output_prefix + '.h5')


def train_network(args):
    # read in potential merge locations
    # TODO fix hardcoding of nanometers
    filename = 'skeletons/' + args.prefix + '_merge_candidates_train_400nm.merge'
    candidates, radii = ReadMergeFilename(filename)

    # read in h5 file
    filename = 'rhoana/' + args.prefix + '_rhoana.h5'
    segmentation = dataIO.ReadH5File(filename, 'main')

    # get the number of candidate locations
    ncandidates = len(candidates)





    import time
    start_time = time.time()

    index = 0
    for epoch in range(1, args.num_epochs + 1):
        if not epoch % 20:
            print '{}/{} in {:4f} seconds'.format(epoch, args.num_epochs, time.time() - start_time)
            start_time = time.time()

        # restart the training examples
        if (index == ncandidates):
            index = 0

        # get the current candidate
        candidate = candidates[index]

        # get the labels for this candidate
        label_one = candidate.label_one
        label_two = candidate.label_two

        # get the position for this candidate
        xposition = candidate.x
        yposition = candidate.y
        zposition = candidate.z

        window = make_window(segmentation, label_one, label_two, xposition, yposition, zposition, radii, args.window_width)
        label = np.zeros((1,1))
        label[:,:] = candidate.ground_truth

        example = np.zeros((1, args.window_width, args.window_width, args.window_width, 1))
        example[0,:,:,:,:] = window

        # TODO clean this up and apply rotation
        current_lr = learning_rate_init / (1.0 + epoch * decay_rate)
        backend.set_value(model.optimizer.lr, current_lr)

        # fir the model
        model.fit(example, label, epochs=1, verbose=0)

        # increment the index
        index += 1

        if not epoch % 500:
            # get the root of the output filename
            root_filename = args.output.split('.')[0]

            json_string = model.to_json()
            open(root_filename + '-' + str(epoch) + '.json', 'w').write(json_string)
            model.save_weights(root_filename + '-' + str(epoch) + '.h5')

    return model

def main():
    # parse the arguments
    parser = argparse.ArgumentParser(description='Train and output a classifier for skeletons')
    parser.add_argument('prefix', help='Prefix for the training dataset')
    parser.add_argument('output', help='Path to save the trained Keras model as an .h5 file')
    #parser.add_argument('--learning_rate', default=0.5, type=float, help='Learning rate for Adadelta optimizer')
#    parser.add_argument('--batch_size', default=5, type=int, help='Batch size to use during training')
    parser.add_argument('--num_epochs', default=2528, type=int, help='Number of epochs in training')
#    parser.add_argument('--epoch_size', default=5, type=int, help='Number of examples per epoch.')
    parser.add_argument('--window_width', default=106, type=int, help='Width of window in each dimension')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='print verbosity')
    args = parser.parse_args()

    # train the model
    model = train_network(args)

    json_string = model.to_json()
    open(args.output.replace('h5', 'json'), 'w').write(json_string)
    model.save_weights(args.output)

if __name__ == '__main__':
    main()
